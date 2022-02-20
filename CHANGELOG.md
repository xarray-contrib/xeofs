# Changelog

<!--next-version-placeholder-->

## v0.3.0 (2022-02-20)
### Feature
* Add Varimax and Promax rotation ([`b42ba16`](https://github.com/nicrie/xeofs/commit/b42ba160f183d7a22a8555b19bf7de340663742b))
* Add Rotator interface for numpy, pandas, xarray ([`050b883`](https://github.com/nicrie/xeofs/commit/050b883113166811bd5f8e6dc35cfcb162fa7503))
* Add varimax and promax algorithms ([`f1e928f`](https://github.com/nicrie/xeofs/commit/f1e928fcb20f2ccfa2f450d2ba45230d01ba1e4c))
* Add Rotator base class ([`d024d81`](https://github.com/nicrie/xeofs/commit/d024d8151429d4bfd6a374207168421ac02242c2))
* Add support for weighted EOF analysis including coslat weighting ([`654b437`](https://github.com/nicrie/xeofs/commit/654b437f64bf5c6dc9be811e891de2c5d1a3d2d9))
* Add weight support to EOF classes ([`8821108`](https://github.com/nicrie/xeofs/commit/882110879a31af5b632efb5a39bf6d6afebe2fb7))
* Add weight transformer ([`52b98e6`](https://github.com/nicrie/xeofs/commit/52b98e6189d144bba4320ceb0dd2c43c1548e8c9))

### Fix
* Incorrect number of mode index for DataArray caller ([`4e610ac`](https://github.com/nicrie/xeofs/commit/4e610aca9b2db726c6351f2615adbb482d011722))
* Always center data X ([`4a58dfc`](https://github.com/nicrie/xeofs/commit/4a58dfc0cc400aa3b20ae0d2c904969d0e19109b))
* Coslat error was too restrictive ([`faece55`](https://github.com/nicrie/xeofs/commit/faece55ccdfaa91f73b6dcce74959dead9736388))
* Add error messages when calling invalid coslat weighting ([`6104e69`](https://github.com/nicrie/xeofs/commit/6104e69b297f42c7aef68e20ca753394fc9a50c8))

### Documentation
* Add example for rotated EOF analysis ([`efc364a`](https://github.com/nicrie/xeofs/commit/efc364a925b33a167bfdfdbb71fd73ebd7b6c6f7))
* Add example for weigted EOF analysis ([`9dedab2`](https://github.com/nicrie/xeofs/commit/9dedab2a25a0f18595e618ca986abe0b57b5a23f))
* Some minor changes in examples ([`9611eea`](https://github.com/nicrie/xeofs/commit/9611eeac466078ac4e008373005e7cd0c98607bd))
* Add EOF s-mode and t-mode gallery example ([`5f371b7`](https://github.com/nicrie/xeofs/commit/5f371b7ee52b64315a8c7940bb993605823e4455))

## v0.2.0 (2022-02-17)
### Feature
* Add support for multidimensional axes ([`7c31c58`](https://github.com/nicrie/xeofs/commit/7c31c58f60376bac57fe42bef58ad9e46942fcb7))

### Fix
* Allow multidimensional axis for decomposition ([`e09a420`](https://github.com/nicrie/xeofs/commit/e09a420561c41c83483ecd1a718d0d6c86ed8c78))

### Documentation
* Add download badge ([`9a96fd1`](https://github.com/nicrie/xeofs/commit/9a96fd1e8d589b4c80b4498224f1851ec0428565))
* Solve readthedoc version issue by installing xeofs first ([`7afdd78`](https://github.com/nicrie/xeofs/commit/7afdd78af786ca5048c748ea09985aecc0d9b7b0))
* Try to solve the readthedocs issue with importlib ([`b4cdd9e`](https://github.com/nicrie/xeofs/commit/b4cdd9ec4ca4d75df9e8a3ba7910163c42970cbe))
* Try to solve readthedoc version number ([`981bcdd`](https://github.com/nicrie/xeofs/commit/981bcdd4865219574bf154bbd6c237c23ee48563))
* Update docstrings ([`e02b6ec`](https://github.com/nicrie/xeofs/commit/e02b6ec4545bc9b13b48f27a00b4da77e1358037))
* Update docs ([`7b19b5b`](https://github.com/nicrie/xeofs/commit/7b19b5bc35564317f49311c1a3705ce0893291dc))
* Add installation instructions ([`43e2563`](https://github.com/nicrie/xeofs/commit/43e2563e986f3217bce6e9fcd643ea0df0297cc4))
* Remove conflicting package versions ([`49636ae`](https://github.com/nicrie/xeofs/commit/49636ae4f456ace63ed19bf081ce2fdf35dbbc42))
* Repair docs due to importlib being installed twice ([`0e21ebd`](https://github.com/nicrie/xeofs/commit/0e21ebd0551ba7813ab5219febfda79dd26aec1a))
* Place badges on same line ([`e2d4dc3`](https://github.com/nicrie/xeofs/commit/e2d4dc380accca197a76c16f815b35f889140150))
* Add installation instruction ([`9512d34`](https://github.com/nicrie/xeofs/commit/9512d3450651384f48582458d2896c4d1ba355cc))

## v0.1.2 (2022-02-15)
### Fix
* Pandas and xarray eofs back_transform was called twice ([`4fa2bfb`](https://github.com/nicrie/xeofs/commit/4fa2bfb3f3a669ad1fd2b8a72f2fb6a64eab927a))
* Allow standardized EOF analysis ([`6e80f78`](https://github.com/nicrie/xeofs/commit/6e80f7867a35079b64a447604701f9e689e63f5f))

### Documentation
* Add batches and link to documentation ([`a7dd2d0`](https://github.com/nicrie/xeofs/commit/a7dd2d0d6cdde42c6c9e9367bfd55d2aa077ba4d))
* Update dependencies ([`05ceb68`](https://github.com/nicrie/xeofs/commit/05ceb68bc77586663d9ddcf36c3e6c42d3947c72))

## v0.1.1 (2022-02-15)
### Fix
* Typo in CI ([`b34ccc5`](https://github.com/nicrie/xeofs/commit/b34ccc511a412dd5920ec6a30d764794ca52aad9))
* Wrong pytest version ([`774b2d6`](https://github.com/nicrie/xeofs/commit/774b2d64af46cc6731e270a25c3e4c524c3d0d94))
* Add development dependencies ([`e1cc1f6`](https://github.com/nicrie/xeofs/commit/e1cc1f669fd218aadf1665b54f441ed1265c6395))
* Add flake8 dependency ([`483cf42`](https://github.com/nicrie/xeofs/commit/483cf4294e5fda29da1477bee073ba552bb40de9))
* Add __version__ ([`739ae74`](https://github.com/nicrie/xeofs/commit/739ae740e8a8f740bd69d73a28daebec7117bcb1))
