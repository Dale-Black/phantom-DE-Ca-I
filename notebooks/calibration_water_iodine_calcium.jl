### A Pluto.jl notebook ###
# v0.14.4

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
filepath = "/Users/daleblack/Google Drive/dev/julia/research/phantom-DE-Ca-I/data/calibration_water_iodine_calcium.csv";

# ╔═╡ be1ca5ec-db9a-48ee-9236-3630f74e7bfb
begin
	df = DataFrame(CSV.File(filepath))
	# Remove first row with only water measurements
	delete!(df, 1)
end

# ╔═╡ 8c483d7a-5a69-4680-bbca-cdc9600c3ace
begin
	df_iodine = df[1:7, :]
	df_calcium = df[8:end, :];
end

# ╔═╡ 687e4bd5-e3ad-4b73-98c1-fcd79bcbd3f3
begin
	train_df_i, test_df_i = MLDataUtils.splitobs(MLDataUtils.shuffleobs(df_iodine), at = 0.8)
	train_df_ca, test_df_ca = MLDataUtils.splitobs(MLDataUtils.shuffleobs(df_calcium), at = 0.8)
end

# ╔═╡ 4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
md"""
## Calibrate
"""

# ╔═╡ a696a524-b9ce-4ab3-a564-027ddafdc7b7
@. multimodel(x, p) = (p[1] + (p[2] * x[:, 1]) + (p[3] * x[:, 2]) + (p[4] * x[:, 1]^2) + (p[5] * x[:, 1] * x[:, 2]) + (p[6] * x[:, 2]^2)) / (1 + (p[7] * x[:, 1]) + (p[8] * x[:, 2]))

# ╔═╡ d4cace91-6695-4d40-ade8-37e7a7014bc4
begin
	xdata_i = hcat(train_df_i[!, :low_energy], train_df_i[!, :high_energy])
	xdata_ca = hcat(train_df_ca[!, :low_energy], train_df_ca[!, :high_energy])
	p0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
end;

# ╔═╡ addca25f-d566-4ff1-bf34-e687bd4840e7
begin
	fit_iodine = LsqFit.curve_fit(multimodel, xdata_i, train_df_i[!, :iodine], p0).param
	
	fit_calcium = LsqFit.curve_fit(multimodel, xdata_ca, train_df_ca[!, :calcium], p0).param
end;

# ╔═╡ 7553f6b0-4f11-42fd-93ee-e96bddaaabce
md"""
## Save fitted params
"""

# ╔═╡ 60b8894f-ba15-4cae-8e78-a81e00606399
params_df = DataFrame("fit_iodine" => fit_iodine, "fit_calcium" => fit_calcium)

# ╔═╡ 4a3b964b-d26e-4226-9b68-30a95e7f06fc
CSV.write("/Users/daleblack/Google Drive/dev/julia/research/phantom-DE-Ca-I/data/params_iodine_calcium1.csv", params_df);

# ╔═╡ 7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
md"""
## Validate
"""

# ╔═╡ eb4a0792-31ba-480a-af99-421837fee8fd
function predict_concentration(x, y, p)
	A = p[1] + (p[2] * x) + (p[3] * y) + (p[4] * x^2) + (p[5] * x * y) + (p[6] * y^2)
	B = 1 + (p[7] * x) + (p[8] * y)
	F = A / B
end

# ╔═╡ 91b6b97e-3575-42b7-b921-34d2859d214c
begin
	arr_iodine = Array{Float64}(undef, nrow(test_df_i))
	for i in 1:nrow(test_df_i)
		arr_iodine[i] =  predict_concentration(test_df_i[!, :low_energy][i], test_df_i[!, :high_energy][i], fit_iodine)
	end
	
	arr_calcium = Array{Float64}(undef, nrow(test_df_ca))
	for i in 1:nrow(test_df_ca)
		arr_calcium[i] =  predict_concentration(test_df_ca[!, :low_energy][i], test_df_ca[!, :high_energy][i], fit_calcium)
	end
end

# ╔═╡ 8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
md"""
Double check that the held out iodine and calcium concentrations seem reasonable by using the calibrated equation. Once calibrated, the equation can predict the concentration given two intensity measurements `low_energy` and `high_energy`
"""

# ╔═╡ 2ccacff3-eee3-45da-8e7c-0689871442d5
test_df_i

# ╔═╡ fbf5b3cd-ac99-4154-a289-941bb7e5030a
arr_iodine

# ╔═╡ d46453a2-3e64-4536-8349-dbd319789955
test_df_ca

# ╔═╡ 5e53d942-64ad-4519-aaa4-cdbc53c3cf44
arr_calcium

# ╔═╡ Cell order:
# ╠═f2fa7d36-9b26-11eb-00e1-3b8d81722ef9
# ╠═9143fafb-5db9-4fd1-8a40-12efb262ea7f
# ╟─fbb7d149-6669-47d4-9a5d-4383dcd40699
# ╟─3e506496-ef3e-4f1e-8968-fb4ed4a50746
# ╠═f12f7207-66d2-46b7-9d18-724b6ea11ca5
# ╠═be1ca5ec-db9a-48ee-9236-3630f74e7bfb
# ╠═8c483d7a-5a69-4680-bbca-cdc9600c3ace
# ╠═687e4bd5-e3ad-4b73-98c1-fcd79bcbd3f3
# ╟─4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
# ╠═a696a524-b9ce-4ab3-a564-027ddafdc7b7
# ╠═d4cace91-6695-4d40-ade8-37e7a7014bc4
# ╠═addca25f-d566-4ff1-bf34-e687bd4840e7
# ╟─7553f6b0-4f11-42fd-93ee-e96bddaaabce
# ╠═60b8894f-ba15-4cae-8e78-a81e00606399
# ╠═4a3b964b-d26e-4226-9b68-30a95e7f06fc
# ╟─7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
# ╠═eb4a0792-31ba-480a-af99-421837fee8fd
# ╠═91b6b97e-3575-42b7-b921-34d2859d214c
# ╟─8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
# ╠═2ccacff3-eee3-45da-8e7c-0689871442d5
# ╠═fbf5b3cd-ac99-4154-a289-941bb7e5030a
# ╠═d46453a2-3e64-4536-8349-dbd319789955
# ╠═5e53d942-64ad-4519-aaa4-cdbc53c3cf44
