"""Phase 3.8 transform, native-grid, frozen-path and raster regression tests."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/phase3_optical/export_optical_phase38.py'
spec = importlib.util.spec_from_file_location('phase38', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_georeferenced_stage_grid_follows_source_window():
    # 10 m geotransform, patch begins 12 rows down and 30 columns right.
    src = Affine(10, 0, 500000, 0, -10, 3400000)
    for stage, stride in enumerate(module.STAGE_STRIDES):
        tx = module.patch_transform(src, 12, 30, stride)
        x, y = tx * (0, 0)
        assert (x, y) == (500300, 3399880)
        assert tx.a == 10*stride and tx.e == -10*stride
        assert (128 // stride) * tx.a == 1280


def test_validity_is_aggregated_without_interpolating_data():
    mask = np.zeros((128,128), dtype=bool)
    mask[:8,:8] = True
    frac = module.cell_valid_fraction(mask,8)
    assert frac.shape == (16,16)
    assert frac[0,0] == 1.0
    assert frac[0,1] == 0.0
    assert frac[1,0] == 0.0


def test_native_stage_tensor_no_spatial_upsampling():
    for idx, (ch,stride) in enumerate(zip(module.STAGE_CHANNELS, module.STAGE_STRIDES)):
        g=128//stride
        x=torch.ones((g,g,ch))
        z=module.native_stage_tensor(x,idx)
        assert z.shape==(ch,g,g)
    with pytest.raises(RuntimeError):
        module.native_stage_tensor(torch.ones((8,8,96)),2)


def test_write_geotiff_retains_transform_crs_and_invalid_mask(tmp_path):
    p=tmp_path/'test.tif'
    tx=Affine(40,0,501000,0,-40,3400000)
    values=np.ones((3,32,32),dtype=np.float32)
    valid=np.ones((32,32),dtype=bool)
    valid[1,2]=False
    module.write_geotiff(p,values,tx,'EPSG:32636',valid,tags={'phase':'3.8'})
    with rasterio.open(p) as src:
        assert src.count==3
        assert src.transform==tx
        assert src.crs.to_epsg()==32636
        assert src.nodata==module.NODATA
        assert src.read(1)[1,2]==module.NODATA
        assert src.dataset_mask()[1,2]==0
        assert src.tags()['phase']=='3.8'


def test_geographic_manifests_never_overlap(tmp_path):
    class FakeDataset:
        samples=[{'patch_id':'x','year':2024,'quarter':1},
                 {'patch_id':'x','year':2024,'quarter':2}]
    assert module.validate_dataset(FakeDataset(),{'test':{'x'},'train':{'a'},'val':{'b'}},'test',2024)==1
    with pytest.raises(RuntimeError):
        module.validate_dataset(FakeDataset(),{'test':{'y'},'train':{'a'},'val':{'b'}},'test',2024)


def test_encoded_full_stage_shapes_on_actual_swin():
    from sarchaeology.models.optical_swin_mim import OpticalSwinMIM
    model=OpticalSwinMIM(depths=(1,1,1,1)).eval()
    with torch.inference_mode():
        z,maps=module.encode_full(model,torch.randn(1,6,128,128))
    assert z.shape==(1,768)
    assert [tuple(x.shape) for x in maps]==[(1,32,32,96),(1,16,16,192),(1,8,8,384),(1,4,4,768)]


def test_refuses_export_into_nonempty_directory(tmp_path):
    (tmp_path/'prior_file').write_text('keep')
    with pytest.raises(FileExistsError):
        module.read_existing_provenance(tmp_path)
