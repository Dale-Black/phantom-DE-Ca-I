### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 3c9403e2-a3c2-11eb-09dd-715f03e071b5
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		
		Pkg.add(Pkg.PackageSpec(; name="CSV"))
		Pkg.add(Pkg.PackageSpec(; name="DataFrames"))
		Pkg.add("PlutoUI")
	end
	
	using CSV
	using DataFrames
	using PlutoUI
end

# ╔═╡ 64f9fffd-0d4f-4350-aad0-9ec2d9833881
TableOfContents()

# ╔═╡ 838492b6-61b2-448e-a1a8-edec89094a91
md"""
# Insert group A
"""

# ╔═╡ 5439fb90-6824-4df0-9362-4fb3d999dddd
md"""
## Input phantom data

Ground truth `Insert group A`:

Given we have 6 inserts with known values of 50 mg/cc calcium per insert and the total volume of calcium for 1 insert is 7.92 mL, then we know the total volume of calcium in total is  47.50 mL. Therefore, we know the total mass of calcium in mg is 2375 mg.

Measured `Insert group A`:

After segmenting the inserts (including both the calcium and the tissue) in 3D slicer, we calculated a mean intensity of `x = 23.0711` for the high energy `135 kV` and `y = 9.3159` for the low energy `80 kV`
"""

# ╔═╡ d75112c2-c1db-4131-a44b-d2482844efdf
begin
	x_measured = 23.0711
	y_measured = 9.3159
end

# ╔═╡ eb5205c9-1304-4f4b-881e-77678c542a26
md"""
## Input fitted calcium and iodine parameters
"""

# ╔═╡ 73c02e49-df6f-4188-a9b4-58328d69c0d5
filepath = "/Users/daleblack/Google Drive/dev/julia/research/phantom-DE-Ca-I/data/params_iodine_calcium1.csv";

# ╔═╡ ce391129-459b-4222-8bf2-a9434a6f982c
df = DataFrame(CSV.File(filepath))

# ╔═╡ ea1c4f2d-7236-4304-9bc3-89c7f37622df
fit_calcium = df[!, :fit_calcium]

# ╔═╡ 367c5b69-a0cd-4f51-8788-ea0f3a5947d5
md"""
## Test using material decomposition
"""

# ╔═╡ d8051222-9f89-48bb-98ac-0427cdef9fb0
function predict_concentration(x, y, p)
	A = p[1] + (p[2] * x) + (p[3] * y) + (p[4] * x^2) + (p[5] * x * y) + (p[6] * y^2)
	B = 1 + (p[7] * x) + (p[8] * y)
	F = A / B
end

# ╔═╡ 08bb670b-97f0-4ac5-8c7b-20a04d4a3dc2
predict_concentration(x_measured, y_measured, fit_calcium)

# ╔═╡ Cell order:
# ╠═3c9403e2-a3c2-11eb-09dd-715f03e071b5
# ╠═64f9fffd-0d4f-4350-aad0-9ec2d9833881
# ╟─838492b6-61b2-448e-a1a8-edec89094a91
# ╟─5439fb90-6824-4df0-9362-4fb3d999dddd
# ╠═d75112c2-c1db-4131-a44b-d2482844efdf
# ╟─eb5205c9-1304-4f4b-881e-77678c542a26
# ╠═73c02e49-df6f-4188-a9b4-58328d69c0d5
# ╠═ce391129-459b-4222-8bf2-a9434a6f982c
# ╠═ea1c4f2d-7236-4304-9bc3-89c7f37622df
# ╟─367c5b69-a0cd-4f51-8788-ea0f3a5947d5
# ╠═d8051222-9f89-48bb-98ac-0427cdef9fb0
# ╠═08bb670b-97f0-4ac5-8c7b-20a04d4a3dc2
