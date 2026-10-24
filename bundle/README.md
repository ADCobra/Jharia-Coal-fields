# Bundled runs

`bundle/real/` holds a processed run of the real Landsat scenes (`blobs/` and `state.db`). When it exists,
`firewatch serve` mounts it read-only under `/real`, next to the simulated demo, and the Docker image carries it.

It is generated, not committed. Make it with:

```bash
bash get_real_data.sh             # download the scenes (data/real/scenes, about 15 MB each)
bash scripts/make_real_bundle.sh  # process them and copy the result here
```
