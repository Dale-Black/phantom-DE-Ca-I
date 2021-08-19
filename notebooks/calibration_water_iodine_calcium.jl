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
		Pkg.add("GalacticOptim")
		Pkg.add("Optim")
		Pkg.add("ModelingToolkit")
	end
	
	using CSV
	using DataFrames
	using LsqFit
	using Statistics
	using MLDataUtils
	using PlutoUI
	using StatsBase
	using NIfTI
	using Optim
	using GalacticOptim
	using ModelingToolkit
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

# ╔═╡ e6fa6e98-2039-4459-9a25-795f656dc851
md"""
# Test other solvers
"""

# ╔═╡ 8e37261a-f40a-47d4-a2b4-39eb6774347e
md"""
```math
\begin{aligned}
	A &= a_o + a_1x + a_2y + a_3x^2 + a_4xy + a_5y^2 \\
	B &= 1 + b_1x + b_2y \\
	F &= \frac{A}{B}
\end{aligned}
```
"""

# ╔═╡ c2b1520d-640c-42f2-8e1a-673e41dc39a2
md"""
### Optim
"""

# ╔═╡ 27090be5-29e9-4a28-939c-204346641284
function equation(p, x)
	p[1] + (p[2] * x[1]) + (p[3] * x[2]) + (p[4] * x[1]^2) + (p[5] * x[1] * x[2]) + (p[6] * x[2]^2) / 1 + (p[7] * x[1]) + (p[8] * x[2])
end

# ╔═╡ 33977ed5-9eb0-4081-9f0c-24f62f0632dc
x0 = [0, 0]

# ╔═╡ ef67495f-c2dc-4d95-8619-6753523d8326
begin
	prob = OptimizationProblem(equation, p0, x0)
	sol = solve(prob, NelderMead())
end;

# ╔═╡ 5964fe9e-9df1-4d7a-a9a7-da89427346bd
with_terminal() do
	@show sol
end

# ╔═╡ 39e03dcd-5ece-4a6b-be24-b08907b973fb
md"""
### ModelingToolkit
"""

# ╔═╡ a0feaaf4-0c25-4fcd-95ea-256a7edb8c95
begin
	@variables x y
	@parameters a₀, a₁, a₂, a₃, a₄, a₅, b₁, b₂
end

# ╔═╡ 63012029-3568-4040-9ea7-cc250c1d0576
begin
	A = a₀ + a₁*x + a₂*y + a₃*x^2 + a₄*x*y + a₅*y^2
	B = 1 + b₁*x + b₂*y
	F = A / B
end

# ╔═╡ 2d4e15bd-eddb-401a-8128-2d36ab8c6262
sys = OptimizationSystem(F, [x, y], [a₀, a₁, a₂, a₃, a₄, a₅, b₁, b₂]);

# ╔═╡ 2494068c-efae-47dd-be4a-da8c2712dfb2
begin
	u0 = [
		x => 1.0
		y => 1.0
	]
	p = [
		a₀ => 1.0
		a₁ => 1.0
		a₂ => 1.0
		a₃ => 1.0
		a₄ => 1.0
		a₅ => 1.0
		b₁ => 1.0
		b₂ => 1.0
	]
end

# ╔═╡ ca096ed7-1ded-4c3c-a574-82f1b1ca78c0
prob2 = OptimizationProblem(sys, u0, p, grad=true, hess=true)

# ╔═╡ 82c2af6a-0251-47ef-8298-d966bd09ff4d
sol2 = solve(prob2, Newton());

# ╔═╡ ecbf2403-07e9-44fc-8bbb-06fa4a3b52b1
with_terminal() do
	@show sol2
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
# ╟─e6fa6e98-2039-4459-9a25-795f656dc851
# ╟─8e37261a-f40a-47d4-a2b4-39eb6774347e
# ╟─c2b1520d-640c-42f2-8e1a-673e41dc39a2
# ╠═27090be5-29e9-4a28-939c-204346641284
# ╠═33977ed5-9eb0-4081-9f0c-24f62f0632dc
# ╠═ef67495f-c2dc-4d95-8619-6753523d8326
# ╠═5964fe9e-9df1-4d7a-a9a7-da89427346bd
# ╟─39e03dcd-5ece-4a6b-be24-b08907b973fb
# ╠═a0feaaf4-0c25-4fcd-95ea-256a7edb8c95
# ╠═63012029-3568-4040-9ea7-cc250c1d0576
# ╠═2d4e15bd-eddb-401a-8128-2d36ab8c6262
# ╠═2494068c-efae-47dd-be4a-da8c2712dfb2
# ╠═ca096ed7-1ded-4c3c-a574-82f1b1ca78c0
# ╠═82c2af6a-0251-47ef-8298-d966bd09ff4d
# ╠═ecbf2403-07e9-44fc-8bbb-06fa4a3b52b1
