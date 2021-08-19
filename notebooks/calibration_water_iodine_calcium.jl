### A Pluto.jl notebook ###
# v0.15.1

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
df = DataFrame(CSV.File(filepath))

# ╔═╡ bc30c569-138e-46a3-8602-1788191ec936
train_df, test_df = splitobs(shuffleobs(df), at = 0.8)

# ╔═╡ 4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
md"""
## Calibrate
"""

# ╔═╡ a696a524-b9ce-4ab3-a564-027ddafdc7b7
multimodel(x, p) = (p[1] .+ (p[2] .* x[:, 1]) .+ (p[3] .* x[:, 2]) .+ (p[4] .* x[:, 1].^2) .+ (p[5] .* x[:, 1] .* x[:, 2]) .+ (p[6] .* x[:, 2].^2)) ./ (1 .+ (p[7] .* x[:, 1]) + (p[8] .* x[:, 2]))

# ╔═╡ 5342b070-8c9b-4bcd-a06a-18cbdb57c1ff
xdata = hcat(train_df[!, :low_energy], train_df[!, :high_energy]);

# ╔═╡ 3ac4340f-1497-4ebe-8015-f6e2e728c655
p0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# ╔═╡ a1041244-5856-4d81-9b3c-cc66e8a64f43
fit_all_ca = LsqFit.curve_fit(multimodel, xdata, train_df[!, :calcium], p0).param

# ╔═╡ e8199f02-554d-473e-9d0e-80cd1acfaec4
fit_all_i = LsqFit.curve_fit(multimodel, xdata, train_df[!, :iodine], p0).param

# ╔═╡ 7553f6b0-4f11-42fd-93ee-e96bddaaabce
md"""
## Save fitted params
"""

# ╔═╡ 60b8894f-ba15-4cae-8e78-a81e00606399
params_df = DataFrame("fit_iodine" => fit_all_i, "fit_calcium" => fit_all_ca)

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

# ╔═╡ cdc842d3-d692-4ac9-8f75-1aa67109a65b
begin
	all_arr_iodine = Array{Float64}(undef, nrow(test_df))
	for i in 1:nrow(test_df)
		all_arr_iodine[i] =  predict_concentration(test_df[!, :low_energy][i], test_df[!, :high_energy][i], fit_all_i)
	end
	
	all_arr_calcium = Array{Float64}(undef, nrow(test_df))
	for i in 1:nrow(test_df)
		all_arr_calcium[i] =  predict_concentration(test_df[!, :low_energy][i], test_df[!, :high_energy][i], fit_all_ca)
	end
end

# ╔═╡ 8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
md"""
Double check that the held out iodine and calcium concentrations seem reasonable by using the calibrated equation. Once calibrated, the equation can predict the concentration given two intensity measurements `low_energy` and `high_energy`
"""

# ╔═╡ 903edf44-e444-4f2d-a61c-15f2decf890a
md"""
### Iodine results
"""

# ╔═╡ 875e06ce-012a-43e6-90bc-5558190ac66b
test_df

# ╔═╡ 48722c03-0ad8-4e03-a44e-2d156cb853aa
all_arr_iodine

# ╔═╡ 7dbd984c-307c-41d8-b260-f6e0158b14bc
md"""
### Calcium results
"""

# ╔═╡ bf048d05-e491-4b94-82d7-387536bc433a
test_df

# ╔═╡ 63a28676-a73a-4088-9969-536f814be3e7
all_arr_calcium

# ╔═╡ Cell order:
# ╠═f2fa7d36-9b26-11eb-00e1-3b8d81722ef9
# ╠═9143fafb-5db9-4fd1-8a40-12efb262ea7f
# ╟─fbb7d149-6669-47d4-9a5d-4383dcd40699
# ╟─3e506496-ef3e-4f1e-8968-fb4ed4a50746
# ╠═f12f7207-66d2-46b7-9d18-724b6ea11ca5
# ╠═be1ca5ec-db9a-48ee-9236-3630f74e7bfb
# ╠═bc30c569-138e-46a3-8602-1788191ec936
# ╟─4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
# ╠═a696a524-b9ce-4ab3-a564-027ddafdc7b7
# ╠═5342b070-8c9b-4bcd-a06a-18cbdb57c1ff
# ╠═3ac4340f-1497-4ebe-8015-f6e2e728c655
# ╠═a1041244-5856-4d81-9b3c-cc66e8a64f43
# ╠═e8199f02-554d-473e-9d0e-80cd1acfaec4
# ╟─7553f6b0-4f11-42fd-93ee-e96bddaaabce
# ╠═60b8894f-ba15-4cae-8e78-a81e00606399
# ╠═4a3b964b-d26e-4226-9b68-30a95e7f06fc
# ╟─7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
# ╠═eb4a0792-31ba-480a-af99-421837fee8fd
# ╠═cdc842d3-d692-4ac9-8f75-1aa67109a65b
# ╟─8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
# ╟─903edf44-e444-4f2d-a61c-15f2decf890a
# ╠═875e06ce-012a-43e6-90bc-5558190ac66b
# ╠═48722c03-0ad8-4e03-a44e-2d156cb853aa
# ╟─7dbd984c-307c-41d8-b260-f6e0158b14bc
# ╠═bf048d05-e491-4b94-82d7-387536bc433a
# ╠═63a28676-a73a-4088-9969-536f814be3e7
