### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ f2fa7d36-9b26-11eb-00e1-3b8d81722ef9
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		
		Pkg.add(Pkg.PackageSpec(; name="CSV"))
		Pkg.add(Pkg.PackageSpec(; name="DataFrames"))
		Pkg.add(Pkg.PackageSpec(; name="LsqFit"))
		Pkg.add(Pkg.PackageSpec(; name="XLSX"))
		Pkg.add(Pkg.PackageSpec(; name="MLDataUtils"))
		Pkg.add("PlutoUI")
		Pkg.add("NIfTI")
	end
	
	using CSV
	using DataFrames
	using LsqFit
	using Statistics
	using MLDataUtils
	using PlutoUI
	using StatsBase
	using NIfTI
end

# ╔═╡ 9143fafb-5db9-4fd1-8a40-12efb262ea7f
TableOfContents()

# ╔═╡ fbb7d149-6669-47d4-9a5d-4383dcd40699
md"""

## Calibration Equation

Let's apply the calibration formula found in [An accurate method for direct dual-energy calibration and decomposition](https://www.researchgate.net/publication/20771282_An_accurate_method_for_direct_dual-energy_calibration_and_decomposition)

```math
A = a_o + a_1x + a_2y + a_3x^2 + a_4xy + a_5y^2
\tag{1}
```

```math
B = 1 + b_1x + b_2y
\tag{2}
```

```math
F = \frac{A}{B}
\tag{3}
```

"""

# ╔═╡ 3e506496-ef3e-4f1e-8968-fb4ed4a50746
md"""

## Load Data

"""

# ╔═╡ f12f7207-66d2-46b7-9d18-724b6ea11ca5
filepath = "/Users/daleblack/Google Drive/dev/julia/DualEnergyCT/data/calibration_water_iodine_calcium - Sheet1.csv";

# ╔═╡ be1ca5ec-db9a-48ee-9236-3630f74e7bfb
df = DataFrame(CSV.File(filepath))

# ╔═╡ 687e4bd5-e3ad-4b73-98c1-fcd79bcbd3f3
train_df, test_df = MLDataUtils.splitobs(MLDataUtils.shuffleobs(df), at = 0.8);

# ╔═╡ 4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
md"""
## Calibrate
"""

# ╔═╡ a696a524-b9ce-4ab3-a564-027ddafdc7b7
@. multimodel(x, p) = (p[1] + (p[2] * x[:, 1]) + (p[3] * x[:, 2]) + (p[4] * x[:, 1]^2) + (p[5] * x[:, 1] * x[:, 2]) + (p[6] * x[:, 2]^2)) / (1 + (p[7] * x[:, 1]) + (p[8] * x[:, 2]))

# ╔═╡ d4cace91-6695-4d40-ade8-37e7a7014bc4
begin
	xdata = hcat(train_df[!, :low_energy], train_df[!, :high_energy])
	p0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
end;

# ╔═╡ addca25f-d566-4ff1-bf34-e687bd4840e7
begin
	fit_water = LsqFit.curve_fit(multimodel, xdata, train_df[!, :percent_water], p0).param
	
	fit_iodine = LsqFit.curve_fit(multimodel, xdata, train_df[!, :percent_iodine], p0).param
	
	fit_calcium = LsqFit.curve_fit(multimodel, xdata, train_df[!, :percent_calcium], p0).param
end;

# ╔═╡ 7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
md"""
## Validate
"""

# ╔═╡ eb4a0792-31ba-480a-af99-421837fee8fd
function calibration(x, y, p)
	A = p[1] + (p[2] * x) + (p[3] * y) + (p[4] * x^2) + (p[5] * x * y) + (p[6] * y^2)
	B = 1 + (p[7] * x) + (p[8] * y)
	F = A / B
end

# ╔═╡ 91b6b97e-3575-42b7-b921-34d2859d214c
begin
	arr_water = Array{Float64}(undef, nrow(test_df))
	for i in 1:nrow(test_df)
		arr_water[i] =  calibration(test_df[!, :low_energy][i], test_df[!, :low_energy][i], fit_water)
	end

	arr_iodine = Array{Float64}(undef, nrow(test_df))
	for i in 1:nrow(test_df)
		arr_iodine[i] =  calibration(test_df[!, :low_energy][i], test_df[!, :low_energy][i], fit_iodine)
	end
	
	arr_calcium = Array{Float64}(undef, nrow(test_df))
	for i in 1:nrow(test_df)
		arr_calcium[i] =  calibration(test_df[!, :low_energy][i], test_df[!, :low_energy][i], fit_calcium)
	end
end

# ╔═╡ c0cbad52-4704-4b47-ba00-9bf7465b22b3
test_df

# ╔═╡ 5e53d942-64ad-4519-aaa4-cdbc53c3cf44
arr_water, arr_iodine, arr_calcium

# ╔═╡ 68a29899-3e99-48ea-8c4d-e8cbd3a1ce99
StatsBase.rmsd(test_df[!, :percent_water], arr_water)

# ╔═╡ 0f0788d3-d494-47ad-92bc-d3dc74f0faf8
StatsBase.rmsd(test_df[!, :percent_iodine], arr_iodine)

# ╔═╡ bb9fdcc7-6407-45a2-95f1-b88cda86d507
StatsBase.rmsd(test_df[!, :percent_calcium], arr_calcium)

# ╔═╡ a6dd23d6-0961-4b71-9028-0d19a12902b5
md"""
## Load phantom data
"""

# ╔═╡ 99ace81e-96ac-4bcd-92b4-55551734c187
niipath = "/Volumes/NeptuneData/UserFolder/Dale/phantom - calcium:iodine - dual energy/scans/STU/VESSEL_SEGMENT/L_5_0_nii/1000_vessel_-_5_mm.nii.gz"

# ╔═╡ 93762389-c2a2-4b6c-9cb8-c9736f7a4fc3
vol = NIfTI.niread(niipath).raw;

# ╔═╡ a215bafb-e4ad-4a60-9aa2-b9d1ee047f1c
vol[vol .< 1024] .= 0

# ╔═╡ Cell order:
# ╠═f2fa7d36-9b26-11eb-00e1-3b8d81722ef9
# ╠═9143fafb-5db9-4fd1-8a40-12efb262ea7f
# ╟─fbb7d149-6669-47d4-9a5d-4383dcd40699
# ╟─3e506496-ef3e-4f1e-8968-fb4ed4a50746
# ╠═f12f7207-66d2-46b7-9d18-724b6ea11ca5
# ╠═be1ca5ec-db9a-48ee-9236-3630f74e7bfb
# ╠═687e4bd5-e3ad-4b73-98c1-fcd79bcbd3f3
# ╟─4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
# ╠═a696a524-b9ce-4ab3-a564-027ddafdc7b7
# ╠═d4cace91-6695-4d40-ade8-37e7a7014bc4
# ╠═addca25f-d566-4ff1-bf34-e687bd4840e7
# ╠═7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
# ╠═eb4a0792-31ba-480a-af99-421837fee8fd
# ╠═91b6b97e-3575-42b7-b921-34d2859d214c
# ╠═c0cbad52-4704-4b47-ba00-9bf7465b22b3
# ╠═5e53d942-64ad-4519-aaa4-cdbc53c3cf44
# ╠═68a29899-3e99-48ea-8c4d-e8cbd3a1ce99
# ╠═0f0788d3-d494-47ad-92bc-d3dc74f0faf8
# ╠═bb9fdcc7-6407-45a2-95f1-b88cda86d507
# ╟─a6dd23d6-0961-4b71-9028-0d19a12902b5
# ╠═99ace81e-96ac-4bcd-92b4-55551734c187
# ╠═93762389-c2a2-4b6c-9cb8-c9736f7a4fc3
# ╠═a215bafb-e4ad-4a60-9aa2-b9d1ee047f1c
