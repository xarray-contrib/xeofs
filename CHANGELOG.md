# Changelog

<!--next-version-placeholder-->

## v0.4.0 (2022-03-02)
### Feature
* Project new data onto EOFs and rotated EOFs ([`d8b0e57`](https://github.com/nicrie/xeofs/commit/d8b0e57622bc6dec1b45ac94821eaf369a335704))
* Project unseen data onto rotated EOFs ([`63b2d3a`](https://github.com/nicrie/xeofs/commit/63b2d3afdcb9b170b3fdbe5d38a6386463423e4a))
* Project unseen data onto EOFs ([`341546b`](https://github.com/nicrie/xeofs/commit/341546b8b74cb1f91105aefd409fab8a087cca9a))
* Project unseen data onto EOFs ([`64e38b1`](https://github.com/nicrie/xeofs/commit/64e38b120a5c7e16431551e4c80f9b4a2a515eb4))
* Allow to reconstruct original data with arbitrary mode combination ([`be095d7`](https://github.com/nicrie/xeofs/commit/be095d77d5d452853a36a6719c7de8edf17bed5b))
* Reconstruct input data after rotation ([`0c9479e`](https://github.com/nicrie/xeofs/commit/0c9479e59a4a016f442b532889437e38c4a0e9bf))
* Reconstruct input data for EOF analysis ([`7ed306a`](https://github.com/nicrie/xeofs/commit/7ed306add5bd7cc9ef9b2e14d486fd7887c1d388))
* Allow different scalings of EOFs an PCs ([`ea39f02`](https://github.com/nicrie/xeofs/commit/ea39f023e1c0cf980063caf2bc2fa7daaac7c8ab))
* Add scaling for PCs and EOFs ([`c2c6fe1`](https://github.com/nicrie/xeofs/commit/c2c6fe190b7a481f3c9193b1ce541c57e3a80e94))
* Add eofs as correlation ([`85960ab`](https://github.com/nicrie/xeofs/commit/85960abf96283978748e283053175577211ade74))
* Eofs as correlation for rotated EOF analysis ([`cb8c472`](https://github.com/nicrie/xeofs/commit/cb8c472f12906d8b2d2750847b1ae62a741fb4f8))
* Eofs as correlation for EOF analysis ([`e53d449`](https://github.com/nicrie/xeofs/commit/e53d4494c96b6335911a79c325382ddc0a57fae4))

### Fix
* Fix incorrect dof for rotated PC scaling ([`addeb82`](https://github.com/nicrie/xeofs/commit/addeb82b0c68f5ffbd6c3f9559503cf88c1ba525))
* PC projections was missing -1 correction for degrees of freedom ([`a243a26`](https://github.com/nicrie/xeofs/commit/a243a26cce09d29b318cb28011e815916f25c2e4))
* Back_transform automatically add feature coords ([`0fef30d`](https://github.com/nicrie/xeofs/commit/0fef30da1bfea0d5b26070474fbe2ee826997dd4))

### Documentation
* Update README ([`982d7e3`](https://github.com/nicrie/xeofs/commit/982d7e3520937b4b696beaa5a4753267a2278280))
* Update README ([`c52763b`](https://github.com/nicrie/xeofs/commit/c52763bbdb4de3f261d996db47125cf44edb6113))
* Update README ([`2d00a71`](https://github.com/nicrie/xeofs/commit/2d00a7126f5248dd766815071857e5c1af63bd28))
* Update README ([`8c8cb29`](https://github.com/nicrie/xeofs/commit/8c8cb29a52496302fa2893f74aa05a9d855fb005))
* Update README ([`58f539b`](https://github.com/nicrie/xeofs/commit/58f539b2d353875d3a3d6da7707f4a1b69079755))
* Add project_onto_eofs to autosummary ([`af7d1f2`](https://github.com/nicrie/xeofs/commit/af7d1f29a33e0e782c9f1cc58932f95f729ee1a6))
* Update docs ([`28e248b`](https://github.com/nicrie/xeofs/commit/28e248b26b840e487370bf7d33ab73fb6b445ce4))
* Add eofs as correlations ([`64c60c1`](https://github.com/nicrie/xeofs/commit/64c60c136ba39805ac9c4886f2f635efdc1e7eb4))
* Update README ([`29f1b4d`](https://github.com/nicrie/xeofs/commit/29f1b4d7c592038d9402ba68fe61cd94b9f72045))
* Remove older version of sphinx-gallery ([`938f294`](https://github.com/nicrie/xeofs/commit/938f2947a91074ebafb4d031403d5c7b2ee3e539))
* Too many "install" ;) ([`ea66ba6`](https://github.com/nicrie/xeofs/commit/ea66ba65be9a33fa99d6b648cec5fc69cde64b85))
* Forgot to specifiy master branch ([`2c827ba`](https://github.com/nicrie/xeofs/commit/2c827ba0e73526cd711f280911025807d2e40837))
* Install current master branch of sphinx-gallery ([`8426033`](https://github.com/nicrie/xeofs/commit/8426033b89b01bac1154532d82967f07c694db42))
* Update links to examples ([`44a4353`](https://github.com/nicrie/xeofs/commit/44a4353c648080aedaa62701d1efba7f757b3e32))
* Add matplotlib to environment to generate example ([`2346fcb`](https://github.com/nicrie/xeofs/commit/2346fcb0b2f8b4b4c62d3bd87891ed107914634c))
* Update examples ([`5795ffa`](https://github.com/nicrie/xeofs/commit/5795ffa0e6902abb536c8912f7b55874b9a141b6))

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
