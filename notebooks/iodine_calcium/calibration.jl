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
		
		Pkg.add([
				"CSV"
				"DataFrames"
				"LsqFit"
				"MLDataUtils"
				"PlutoUI"
				])
	end
	
	using CSV
	using DataFrames
	using LsqFit
	using Statistics
	using MLDataUtils
	using PlutoUI
	using StatsBase
end

# ╔═╡ 9143fafb-5db9-4fd1-8a40-12efb262ea7f
TableOfContents()

# ╔═╡ fbb7d149-6669-47d4-9a5d-4383dcd40699
md"""

## Calibration Equation

Let's apply the calibration formula found in [An accurate method for direct dual-energy calibration and decomposition](https://www.researchgate.net/publication/20771282_An_accurate_method_for_direct_dual-energy_calibration_and_decomposition)

```math
\begin{aligned}
	A &= a_o + a_1x + a_2y + a_3x^2 + a_4xy + a_5y^2 \\
	B &= 1 + b_1x + b_2y \\
	F &= \frac{A}{B}
\end{aligned}
\tag{1}
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

# ╔═╡ 753c4856-295a-49ac-a27a-b6bd471e3933
md"""
The `:water`, `:iodine`, and `:calcium` values correspond to the calibration insert densities, given in units of ``\frac{mg}{mL}``. The `:low_energy` is 80 kV and the `:high_energy` is 135 kV. The values in the energy columns are simply the mean intensity values of the segmented iodine/calcium calibration inserts
"""

# ╔═╡ bc30c569-138e-46a3-8602-1788191ec936
train_df, test_df = splitobs(shuffleobs(df), at = 0.8)

# ╔═╡ 4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
md"""
## Calibrate
"""

# ╔═╡ a696a524-b9ce-4ab3-a564-027ddafdc7b7
multimodel(x, p) = (p[1] .+ (p[2] .* x[:, 1]) .+ (p[3] .* x[:, 2]) .+ (p[4] .* x[:, 1].^2) .+ (p[5] .* x[:, 1] .* x[:, 2]) .+ (p[6] .* x[:, 2].^2)) ./ (1 .+ (p[7] .* x[:, 1]) + (p[8] .* x[:, 2]))

# ╔═╡ b5c7a7dd-bf43-463a-aa1c-29124a087ca7
md"""
Model needs some starting parameters. These might need to be adjusted to improve the fitting. We can also investigate a better technique
"""

# ╔═╡ 3ac4340f-1497-4ebe-8015-f6e2e728c655
p0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# ╔═╡ 5342b070-8c9b-4bcd-a06a-18cbdb57c1ff
xdata = hcat(train_df[!, :low_energy], train_df[!, :high_energy]);

# ╔═╡ a1041244-5856-4d81-9b3c-cc66e8a64f43
fit_all_ca = LsqFit.curve_fit(multimodel, xdata, train_df[!, :calcium], p0).param

# ╔═╡ e8199f02-554d-473e-9d0e-80cd1acfaec4
fit_all_i = LsqFit.curve_fit(multimodel, xdata, train_df[!, :iodine], p0).param

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
	all_arr_iodine = []
	all_arr_calcium = []
	
	for i in 1:nrow(test_df)
		push!(all_arr_iodine, predict_concentration(
				test_df[!, :low_energy][i], 
				test_df[!, :high_energy][i], 
				fit_all_i))
	end
	
	for i in 1:nrow(test_df)
		push!(all_arr_calcium, predict_concentration(
				test_df[!, :low_energy][i], 
				test_df[!, :high_energy][i], 
				fit_all_ca))
	end
end

# ╔═╡ 903edf44-e444-4f2d-a61c-15f2decf890a
md"""
### Results
"""

# ╔═╡ 8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
md"""
Double check that the held out iodine and calcium concentrations seem reasonable by using the calibrated equation. Once calibrated, the equation can predict the concentration given two intensity measurements `low_energy` and `high_energy`
"""

# ╔═╡ 53efd7b8-3038-4c2e-b033-71b3eebedb3b
begin
	results = copy(test_df)
	results[!, :predicted_iodine] = all_arr_iodine
	results[!, :predicted_calcium] = all_arr_calcium
	results
end

# ╔═╡ 7553f6b0-4f11-42fd-93ee-e96bddaaabce
md"""
## Save fitted params
"""

# ╔═╡ 60b8894f-ba15-4cae-8e78-a81e00606399
params_df = DataFrame("fit_iodine" => fit_all_i, "fit_calcium" => fit_all_ca)

# ╔═╡ 4a3b964b-d26e-4226-9b68-30a95e7f06fc
CSV.write("/Users/daleblack/Google Drive/dev/julia/research/phantom-DE-Ca-I/data/params_iodine_calcium1.csv", params_df);

# ╔═╡ e7c294a1-6ab8-41f7-8c55-7baf84bfc61c
md"""
## Results all
"""

# ╔═╡ b81182e0-c444-4782-bd83-38f32ccc80a0
begin
	all_arr_iodine_all = []
	all_arr_calcium_all = []
	
	for i in 1:nrow(df)
		push!(all_arr_iodine_all, predict_concentration(
				df[!, :low_energy][i], 
				df[!, :high_energy][i], 
				fit_all_i))
	end
	
	for i in 1:nrow(df)
		push!(all_arr_calcium_all, predict_concentration(
				df[!, :low_energy][i], 
				df[!, :high_energy][i], 
				fit_all_ca))
	end
end

# ╔═╡ f206fb12-9481-4f84-a9ef-effc19498067
begin
	results_all = copy(df)
	results_all[!, :predicted_iodine] = all_arr_iodine_all
	results_all[!, :predicted_calcium] = all_arr_calcium_all
	results_all
end

# ╔═╡ Cell order:
# ╠═f2fa7d36-9b26-11eb-00e1-3b8d81722ef9
# ╠═9143fafb-5db9-4fd1-8a40-12efb262ea7f
# ╟─fbb7d149-6669-47d4-9a5d-4383dcd40699
# ╟─3e506496-ef3e-4f1e-8968-fb4ed4a50746
# ╠═f12f7207-66d2-46b7-9d18-724b6ea11ca5
# ╠═be1ca5ec-db9a-48ee-9236-3630f74e7bfb
# ╟─753c4856-295a-49ac-a27a-b6bd471e3933
# ╠═bc30c569-138e-46a3-8602-1788191ec936
# ╟─4689b9a5-5f6a-4333-a76e-8f0a25a2e5b3
# ╠═a696a524-b9ce-4ab3-a564-027ddafdc7b7
# ╟─b5c7a7dd-bf43-463a-aa1c-29124a087ca7
# ╠═3ac4340f-1497-4ebe-8015-f6e2e728c655
# ╠═5342b070-8c9b-4bcd-a06a-18cbdb57c1ff
# ╠═a1041244-5856-4d81-9b3c-cc66e8a64f43
# ╠═e8199f02-554d-473e-9d0e-80cd1acfaec4
# ╟─7d851c3f-9f63-45c0-8c1a-ad5a1fef329f
# ╠═eb4a0792-31ba-480a-af99-421837fee8fd
# ╠═cdc842d3-d692-4ac9-8f75-1aa67109a65b
# ╟─903edf44-e444-4f2d-a61c-15f2decf890a
# ╟─8616b7a0-39a4-4899-ab0c-13ef53d3a3d2
# ╠═53efd7b8-3038-4c2e-b033-71b3eebedb3b
# ╟─7553f6b0-4f11-42fd-93ee-e96bddaaabce
# ╠═60b8894f-ba15-4cae-8e78-a81e00606399
# ╠═4a3b964b-d26e-4226-9b68-30a95e7f06fc
# ╟─e7c294a1-6ab8-41f7-8c55-7baf84bfc61c
# ╠═b81182e0-c444-4782-bd83-38f32ccc80a0
# ╠═f206fb12-9481-4f84-a9ef-effc19498067
