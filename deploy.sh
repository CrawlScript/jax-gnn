#m2r README.md
rm -rf jax_gnn.egg-info
rm -rf dist
python setup.py sdist
twine upload dist/* --verbose
