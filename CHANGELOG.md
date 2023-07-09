# Changelog

<!--next-version-placeholder-->

## v2.0.0 (2023-07-09)

### Feature

* Complex MCA amplitude and phase ([`55ce3b1`](https://github.com/nicrie/xeofs/commit/55ce3b17f2cb77ea2f11e4fe6444f9860ca5920d))
* Add meta data to model output ([`083a8e0`](https://github.com/nicrie/xeofs/commit/083a8e049140bfbec87f354ed7f0504bbb208fd8))
* Skeleton of Bootstrapper class ([`4934b31`](https://github.com/nicrie/xeofs/commit/4934b31f8ab3d2d35f371f13abedfd5c178775a1))
* Rotation supports dask input ([`78360cf`](https://github.com/nicrie/xeofs/commit/78360cfbc3b237e8791a32b65aca3f0e7b5d7ec7))
* Add complex MCA ([`13f8bbc`](https://github.com/nicrie/xeofs/commit/13f8bbc8b29e82af37ec6793b416a0ca1e2d1aa5))
* RotatorFactory to access rotator classes ([`90b2db6`](https://github.com/nicrie/xeofs/commit/90b2db687314bc1b62aae5c74c0817eeb806203e))
* EOF class transform, inverse_trans and corr ([`fb71ffe`](https://github.com/nicrie/xeofs/commit/fb71ffede30fdfd65b4b812a62340e9f292fbea6))
* Add support for complex EOF ([`6bff6af`](https://github.com/nicrie/xeofs/commit/6bff6af12f0202fbce9cf06453ac66e8921d1d5c))

### Fix

* Add dependency statsmodels ([`87e7e1d`](https://github.com/nicrie/xeofs/commit/87e7e1d89f5d8dd3f7954bb4ebc79d2d41738404))
* Add components and pattern method to MCA ([`849059b`](https://github.com/nicrie/xeofs/commit/849059b65d9218753ef886f5790742ea832a504d))
* Merge 'release-v1.0.0' into bootstrapper ([`e6ea275`](https://github.com/nicrie/xeofs/commit/e6ea27536a43ff086c615ed720a03166d20718de))
* Stacker in T-mode ([`2f9be99`](https://github.com/nicrie/xeofs/commit/2f9be995f2a73e75c0bf88b86290246effc5989c))
* Supress warning when unstacking coords ([`2f01695`](https://github.com/nicrie/xeofs/commit/2f01695eac40bae2519f7dfd7b4d936b4c6647c5))
* Number of modes to be rotated defaults 10 ([`b13c833`](https://github.com/nicrie/xeofs/commit/b13c833bd12241878b218cf62bbdc3121a8034de))
* Rename n_rot to n_modes ([`5b39cd4`](https://github.com/nicrie/xeofs/commit/5b39cd4d565a82185c76f16b248e6aeae78577cc))
* N_components instead of n_modes ([`5d282b1`](https://github.com/nicrie/xeofs/commit/5d282b1fc83f150113b13d4f736838676e5d9fff))
* Change parameter dims to dim ([`70fe651`](https://github.com/nicrie/xeofs/commit/70fe65147f2dab8be7d9bdf08a81fbd36cc45897))
* Phase of complex methods returns np.ndarray ([`dfb050d`](https://github.com/nicrie/xeofs/commit/dfb050d82b8d12cc137bd51316b220dd1deb93c3))
* Complex decomposition not used ([`2086546`](https://github.com/nicrie/xeofs/commit/208654683c9071bc3927e8c7dd549a01e409dea3))
* Added missing import ([`207af0a`](https://github.com/nicrie/xeofs/commit/207af0ab15267eca5fccbbfb5a464ceb4004d56e))
* Remove unecessary  dimensions in scores ([`63e2204`](https://github.com/nicrie/xeofs/commit/63e2204ab3cce1fdf4ae6a1a153a987fde69e5c0))
* Reindex data to ensure deterministic output ([`60c382b`](https://github.com/nicrie/xeofs/commit/60c382bc181aacb8997d955c225df96a7b3bed11))
* ListStacker correctly unstacks ([`e363357`](https://github.com/nicrie/xeofs/commit/e363357851199b08d916b8efae4bac6a56f5c806))
* Define names of output arrays ([`c826aa8`](https://github.com/nicrie/xeofs/commit/c826aa81dbd1bc9c6441982847bff08c4e9cd333))

### Breaking

* rename n_rot to n_modes ([`5b39cd4`](https://github.com/nicrie/xeofs/commit/5b39cd4d565a82185c76f16b248e6aeae78577cc))
* n_components instead of n_modes ([`5d282b1`](https://github.com/nicrie/xeofs/commit/5d282b1fc83f150113b13d4f736838676e5d9fff))
* change parameter dims to dim ([`70fe651`](https://github.com/nicrie/xeofs/commit/70fe65147f2dab8be7d9bdf08a81fbd36cc45897))
* drop support for pandas ([`96196e5`](https://github.com/nicrie/xeofs/commit/96196e55a3094ae63266b534aa36e4cedf56d03a))

### Documentation

* Improve documentation ([#48](https://github.com/nicrie/xeofs/issues/48)) ([`378aae8`](https://github.com/nicrie/xeofs/commit/378aae871c15ad19b1e63631d58d8b00bafd65a2))
* Move to pydata sphinx theme ([`9e92920`](https://github.com/nicrie/xeofs/commit/9e92920d75114f1525ab59f01e81daf044f3f975))
* Add comparison to other packages ([`7985585`](https://github.com/nicrie/xeofs/commit/7985585b34fd27fd391f8a0d388723e2f639df30))
* Fix broken badge ([`9d9b5d8`](https://github.com/nicrie/xeofs/commit/9d9b5d889bbef74f67c3a2b9d946c2373e51d725))
* Add dev-dependeny for readthedocs ([#46](https://github.com/nicrie/xeofs/issues/46)) ([`e1e6379`](https://github.com/nicrie/xeofs/commit/e1e6379e2da146e7d8422da45e68bf678561d600))
* Add sphinx related packages to env ([`6e07d3b`](https://github.com/nicrie/xeofs/commit/6e07d3b3c6797a787a5b10885c2f73ef5c14cdf8))
* Improve documentation ([`b7c6680`](https://github.com/nicrie/xeofs/commit/b7c6680d196b269301b16143626fc0fea15cd038))
* Add more docstrings ([`84ebb5a`](https://github.com/nicrie/xeofs/commit/84ebb5ac9a4abca9b30c04b9e9089d3c73ce15a7))
* Add docstrings ([`0fe6e24`](https://github.com/nicrie/xeofs/commit/0fe6e242f9f4bc1067e8b2fb8e2c0eafaaebf2b2))

### Performance

* Always compute scaling arrays prior to analy ([`5b810ce`](https://github.com/nicrie/xeofs/commit/5b810ce3f7ecddd9fb44d307e600cd472e07d599))

## v1.0.1 (2023-07-07)

### Fix

* Build and ci ([#45](https://github.com/nicrie/xeofs/issues/45)) ([`7d1a88b`](https://github.com/nicrie/xeofs/commit/7d1a88b1cda8a66d04f3ffa96e1aa5cfe899029b))
* Add dask as dependency ([#42](https://github.com/nicrie/xeofs/issues/42)) ([`2bb2b6b`](https://github.com/nicrie/xeofs/commit/2bb2b6b817a457a7a24918914e88675f08e298d6))

## v1.0.0 (2023-07-07)

### Feature

* V1.0.0 ([`ec70e8a`](https://github.com/nicrie/xeofs/commit/ec70e8a9321d0aa1dc0b44ca83be14f441afef18))

### Breaking

* drop pandas support; add support for dask, complex EOF and flexible inputs ([`ec70e8a`](https://github.com/nicrie/xeofs/commit/ec70e8a9321d0aa1dc0b44ca83be14f441afef18))

## v0.7.2 (2023-01-10)
### Fix
* FutureWarning in coslat check ([#37](https://github.com/nicrie/xeofs/issues/37)) ([`285fe0f`](https://github.com/nicrie/xeofs/commit/285fe0f6f6cb69cd84e3ac4c662c64d6d659ef47))

## v0.7.1 (2023-01-08)
### Fix
* Allow newer xarray version ([`49723c0`](https://github.com/nicrie/xeofs/commit/49723c0771b87b8f4b812572f51d50f71bb139e3))

## v0.7.0 (2022-08-26)
### Feature
* Add support for ROCK-PCA ([`202844d`](https://github.com/nicrie/xeofs/commit/202844d0e12565bdefb39988a374c4aa20681a0d))
* Merge branch 'develop' into rock-pca ([`6a5bda8`](https://github.com/nicrie/xeofs/commit/6a5bda8ab1fdc3e0c8c2395172385e058c0b7d3d))
* Add ROCK PCA ([`0ba0660`](https://github.com/nicrie/xeofs/commit/0ba0660fa4f2396dc537888c80be5352dedaebc4))
* Add Rotator class for MCA ([`0e9e8f9`](https://github.com/nicrie/xeofs/commit/0e9e8f90f00d499a385956742fc99ca0776bed83))
* Add Rotator class for MCA ([`6adf45f`](https://github.com/nicrie/xeofs/commit/6adf45fe0d4a126726c503ab45469f3e488b4890))

### Fix
* Add stabilizer for communalities during rotation ([`462f2fe`](https://github.com/nicrie/xeofs/commit/462f2fe9b30959076a815f3236b48d94c4467f32))
* Numpy and pandas classes did not consider axis parameter ([`8b75271`](https://github.com/nicrie/xeofs/commit/8b75271be096107f8a670f97ea6afe2d4e9740a9))

### Documentation
* Bibtex not showing up on Github ;) ([`0c2a663`](https://github.com/nicrie/xeofs/commit/0c2a6635ee5942f2c38f28a1f529ec6a4a5e24bd))
* Add bibtex ([`1428ebf`](https://github.com/nicrie/xeofs/commit/1428ebfc5d65a62044a3f9f9fb20a4636dbfb891))
* Fix some minor errors ([`d5d3f73`](https://github.com/nicrie/xeofs/commit/d5d3f73b27814b947903a30cf6cbde8aaf5dc67b))
* Update README ([`2d28995`](https://github.com/nicrie/xeofs/commit/2d28995a9e6c5ce1424721497eef6e97a6430e45))
* Change examples ([`1c69645`](https://github.com/nicrie/xeofs/commit/1c6964542dcfe3d794c6a01442822f57d422a681))
* Adding example for ROCK PCA ([`8c6da93`](https://github.com/nicrie/xeofs/commit/8c6da93f7c6e99780299e2687960c6a22e7c6661))
* Update ROCK PCA to documentation ([`3a7394d`](https://github.com/nicrie/xeofs/commit/3a7394d57fb4e9d79dfffef5b32df5af1a52e179))
* Update README ([`9e3210d`](https://github.com/nicrie/xeofs/commit/9e3210d190da254850ea17c70011dab916bda24c))
* Add example and update docs ([`8bed38a`](https://github.com/nicrie/xeofs/commit/8bed38a79094ece72487b619aa01cd45fa276a80))
* Some minor corrections in docstrings ([`75eed31`](https://github.com/nicrie/xeofs/commit/75eed31f2cdf33a896174aca77c33ec4bc3791eb))
* More text ([`0f9c32e`](https://github.com/nicrie/xeofs/commit/0f9c32e48dd6c9069c11802a13a3f0113e5f07f5))
* Fix docs ([`19bb84e`](https://github.com/nicrie/xeofs/commit/19bb84e3c57c4762fb2d61b3a60df143e6c05b72))

## v0.6.0 (2022-08-22)
### Feature
* Add MCA ([`34a82d1`](https://github.com/nicrie/xeofs/commit/34a82d103699cb1b1607e2418eb3c0889fad96fb))
* Add MCA support for xarray ([`e816e36`](https://github.com/nicrie/xeofs/commit/e816e3699928d19e828fe0bb41b5003bba6a264e))
* Add MCA support for pandas ([`834d7dd`](https://github.com/nicrie/xeofs/commit/834d7dda131ffaf4336f775519f34228ddf62d69))
* Add MCA support for numpy ([`8ded4df`](https://github.com/nicrie/xeofs/commit/8ded4df531281b3e19359a5d26f3e5bf4c2db320))
* Add MCA base class ([`58612e4`](https://github.com/nicrie/xeofs/commit/58612e40ad225ce4ca30757904e5f7836b3202bc))
* Add bootstrap methods ([`d5f6797`](https://github.com/nicrie/xeofs/commit/d5f6797ab087baabcdf71af325b0754bb3495477))
* Bootstrapper for xarray ([`f807ea6`](https://github.com/nicrie/xeofs/commit/f807ea6dd374e989bab0a95f1ac3e5fb0a9dc282))
* Bootstrapper for pandas ([`a32b1d3`](https://github.com/nicrie/xeofs/commit/a32b1d30a33d695b4c49a121fc343d57a68ec3d4))
* Bootstrapper for numpy class ([`c5923b3`](https://github.com/nicrie/xeofs/commit/c5923b3822178f9ad63837ea841dbe408e8cb3f0))
* Bootstrapper base class ([`f4ee31a`](https://github.com/nicrie/xeofs/commit/f4ee31a9fe83637c1a641f6d1d05844ed15c0ba7))

### Fix
* Set informative names of Dataframes and DataArrays ([`b5b5286`](https://github.com/nicrie/xeofs/commit/b5b528678becdf80b511a3883485304341c09692))

### Documentation
* Minor restructuring ([`dbdc885`](https://github.com/nicrie/xeofs/commit/dbdc8850befe142d567181250793202dc0e68c44))
* Remove some old examples ([`625dd08`](https://github.com/nicrie/xeofs/commit/625dd0827cd3bda178b3c83629d399947c1b5877))
* Minor changes in text and example arrangements ([`b7f1628`](https://github.com/nicrie/xeofs/commit/b7f162800f012e816a6243cfe3e321cf7d9d3aeb))
* Update documentation and docstrings ([`b8fffdc`](https://github.com/nicrie/xeofs/commit/b8fffdc32387ed1ceea63674675d2ac437fe85d9))
* Add MCA example ([`4fb881e`](https://github.com/nicrie/xeofs/commit/4fb881edcad9e7171d8045935ef32fa6a87caff0))
* Reorganize examples ([`68d9db0`](https://github.com/nicrie/xeofs/commit/68d9db004ff23574fafb7b69cc85c7b2b33812c0))
* Add figure to bootstrapping example ([`69894a0`](https://github.com/nicrie/xeofs/commit/69894a0363eda7886969ff7544ed069067bf1f51))
* Add docstrings to bootstrapping methods ([`9c8145c`](https://github.com/nicrie/xeofs/commit/9c8145ccd26d1a5150f6c33bb157501cf6d42bca))
* Add simple example for bootstrapping ([`ba62057`](https://github.com/nicrie/xeofs/commit/ba620578b379636a0fff7e914bf753c1c5397f73))
* Add install instructions for conda ([`ef293e5`](https://github.com/nicrie/xeofs/commit/ef293e5a97b294c0aeea070a9b77fa33f214dcdf))
* Add zenodo badge ([`4f338ef`](https://github.com/nicrie/xeofs/commit/4f338ef473ac1e742452f130fef7604d0c33dc5f))

## v0.5.0 (2022-03-12)
### Feature
* Add support for multivariate EOF analysis ([`53961d9`](https://github.com/nicrie/xeofs/commit/53961d974cda8bc6b24466c496058efc4d676a4b))
* Merge branch 'main' into develop ([`6d2d646`](https://github.com/nicrie/xeofs/commit/6d2d6469768d3b91c63358d561b63f9581ebf2a8))
* Add support for multivariate EOF analysis ([`fa9503a`](https://github.com/nicrie/xeofs/commit/fa9503a2a789404471b2d85121d54e575a83128c))
* Add base and xarray class for multivariate EOF analysis ([`5ba07f0`](https://github.com/nicrie/xeofs/commit/5ba07f0e7c211e9b1a19a44d66d85d3ffc30a4d3))

### Documentation
* Update README ([`fdc76ee`](https://github.com/nicrie/xeofs/commit/fdc76ee567d442cc310571b808a6947774f23e06))
* Add example for multivariate EOF analysis ([`59a1f1b`](https://github.com/nicrie/xeofs/commit/59a1f1be37bb1bed5d9288841ddde891c03c7600))
* Add example ([`07b3bb8`](https://github.com/nicrie/xeofs/commit/07b3bb8d72f2f850dfa61e08613954b7c11cc99a))
* Add example for multivariate EOF analysis ([`7ae2ae8`](https://github.com/nicrie/xeofs/commit/7ae2ae8180a0f997c2f31d45eba5daa747c9900d))
* Add zenodo badge ([`7792953`](https://github.com/nicrie/xeofs/commit/7792953e478eeb0e772563999a6ee0688d06ad76))
* Update README ([`5693fe9`](https://github.com/nicrie/xeofs/commit/5693fe9f2e2b10f1d0c364d0aba1eb47c84e9bc9))
* Fix typo ([`d5505c6`](https://github.com/nicrie/xeofs/commit/d5505c6c6e9a010cb836b609ebbf7dac6b38f67e))

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
