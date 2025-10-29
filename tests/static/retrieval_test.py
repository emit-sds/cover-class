import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd # type: ignore[import]
import torch
import io
import contextlib

from cover_class.static import retrieval # type: ignore[import]
MODULE = "cover_class.static.retrieval"


class retrievalTest(unittest.TestCase):
    def test_from_config(self):
        local_path = "local.csv"
        url = "https://example.com/data.csv"
        config = {
            "datasets": {
                "output-directory": "/test",
                "classes": {
                    "soil": [local_path, url],
                    "pv": [local_path, local_path],
                    "npv": [url, url],
                    "ash+char": [url, url],
                },
            },
            "target-wavelengths-file": "/target.npy",
        }

        data = np.array([[0.1, 0.2]], dtype=np.float32)
        target_wl = np.array([350.0, 500.0], dtype=np.float32)
        interp_return = torch.zeros((1,), dtype=torch.float32)

        with contextlib.redirect_stdout(io.StringIO()): # don't print to console
            with patch(f"{MODULE}.read_config", return_value=config), \
                patch(f"{MODULE}.np.load", return_value=target_wl), \
                patch.object(Path, "is_dir"), \
                patch.object(Path, "exists"), \
                patch(f"{MODULE}.vfs_csv", return_value=(target_wl, data)) as mock_vfs, \
                patch(f"{MODULE}.download", return_value=(target_wl, data)) as mock_dl, \
                patch(f"{MODULE}.interior_interpolation",return_value=(interp_return, target_wl)) as mock_interp, \
                patch(f"{MODULE}.make_hdf5", return_value=f"{config['datasets']['output-directory']}/out.hdf5") as mock_h5:
                
                retrieval.generate_hdf5_from_config("dummy_config_path.yaml")
                mock_dl.assert_any_call(url)
                self.assertEqual(len(mock_dl.call_args_list), 8) # due to the mocks, it should be called 8 times
                mock_vfs.assert_not_called()

                with patch.object(Path, "is_file"):
                    retrieval.generate_hdf5_from_config("dummy_config_path.yaml")
                    mock_vfs.assert_any_call(local_path)
                    self.assertEqual(len(mock_vfs.call_args_list), 8)

        self.assertTrue(mock_interp.called)
        self.assertTrue(mock_h5.called)


    def test_download(self):
        wls = [350.1 + (i*10) for i in range(10)]
        wlv = [0.1 for _ in range(len(wls))]
        wlc = ','.join(str(i) for i in wls)
        wlvc = ','.join(str(i) for i in wlv)
        csv_bytes = (f"id,450,500,blah,{wlc},note\nx,0.1,0.2,a,{wlvc},b\n").encode()

        class Resp:
            content = csv_bytes
            def raise_for_status(self): ...
            
        with patch(f"{MODULE}.requests.get", return_value=Resp()) as mock_request:
            wl, sp = retrieval.download("https://example.com/fake.csv")

        mock_request.assert_called_once()
        np.testing.assert_allclose(wl, np.array(wls, dtype=np.float32))
        np.testing.assert_allclose(sp, np.array([wlv], dtype=np.float32))


    def test_vfs_csv(self):
        spectra = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]], dtype=np.float32)
        wls = np.array([450.0, 500.0, 550.0], dtype=np.float32)
        df = pd.DataFrame(
            {
                "meta": ["a", "b"],
                str(wls[0]): spectra[:, 0],
                str(wls[1]): spectra[:, 1],
                str(wls[2]): spectra[:, 2],
                "tail": ["x", "y"],
            }
        )

        with patch(f"{MODULE}.pd.read_csv", return_value=df) as mock_read:
            wl, sp = retrieval.vfs_csv("any/path.csv")

        mock_read.assert_called_once()
        np.testing.assert_allclose(wl, wls)
        np.testing.assert_allclose(sp, spectra)


if __name__ == "__main__":
    unittest.main()
