
ˆ
îsm80_xmma_dgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnnˇ Äê*Ä2…	8∏Ù◊@∏Ù◊H∏Ù◊Xb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputh
ƒ
‚_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE® Ä¿*Ä28»‡ê@»‡êH»‡êXb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterh
Ï
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)¢ Ä¿*Ä2¡8Á¢â@Á¢âHÁ¢âPXbmodel/2_conv2d/Conv2Dh
ƒ
‚_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE® Ä¿*Ä28Ä˙ﬁ@Ä˙ﬁHÄ˙ﬁXb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterh
®
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2¡8ˇΩ“@ˇΩ“HˇΩ“PXb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputh
Ï
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)¢ Ä¿*Ä2ê8µ¨ç@µ¨çHµ¨çPXbmodel/3_conv2d/Conv2Dh
¡
‚_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE® Ä¿*Ä2	8∞Éi@∞ÉiH∞ÉiXb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterh
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8å’V@å’VHå’Vb(gradient_tape/model/2_activation/EluGradhuZUÖB
§
¬void wgrad_alg0_engine_NHWC<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)NÄ*2¿8ÎæE@ÎæEHÎæEXb8gradient_tape/model/1_conv2d/Conv2D/Conv2DBackpropFilterhu  HB
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2ê8Èø9@Èø9HÈø9PXb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)¢ Ä¿*Ä2§8»Œ2@»Œ2H»Œ2PXbmodel/4_conv2d/Conv2Dh
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8¶ˇ+@¶ˇ+H¶ˇ+Xb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhu≥™&B
¢
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8á∏+@á∏+Há∏+Xb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhu≥™&B
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8∆¶+@∆¶+H∆¶+Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu≥™&B
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8Üà+@Üà+HÜà+Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu≥™&B
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8Ê˝*@Ê˝*HÊ˝*Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhu≥™&B
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	¿8á÷*@á÷*Há÷*Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhu≥™&B
„
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8≈ò'@Ä4H≈‰&Xb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputhu  »B
È
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8Êî%@Êî%HÊî%b/model/2_DO/dropout/random_uniform/RandomUniformhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208Ñƒ@ÑƒHÑƒb0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZUÖB
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2ÄM8ƒ∫@¡‚HÇÏb-gradient_tape/model/1_BN/FusedBatchNormGradV3hu  »B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ò8„Á@„ÁH„Ábmodel/5_DO/dropout/GreaterEqualhuZUÖB
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8√@√H√b-gradient_tape/model/1_BN/FusedBatchNormGradV3hu  »B
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿<8ÉÓ@°ÔH°Öb-gradient_tape/model/2_BN/FusedBatchNormGradV3hu  »B
Ô
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8Éÿ@ÉÿHÉÿb)gradient_tape/model/activation_3/ReluGradhuZUÖB
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2	†8Çô@ÇôHÇôXb8gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropFilterhu≥™&B
Á
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8£á@£áH£áPXbmodel/conv2d/Conv2Dh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8‚Ç@‚ÇH‚ÇPXbmodel/conv2d_5/Conv2Dh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8‚¸@‚¸H‚¸PXbmodel/conv2d_4/Conv2Dh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8É¯@É¯HÉ¯PXbmodel/conv2d_2/Conv2Dh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8É˜@É˜HÉ˜PXbmodel/conv2d_3/Conv2Dh
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2§8£Û@£ÛH£ÛPXbmodel/conv2d_1/Conv2Dh
£
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8ÉÓ@ÉÓHÉÓPXb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputh
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8ÉÎ@ÉÎHÉÎPXb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputh
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8¬‚@¬‚H¬‚PXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputh
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8£›@£›H£›PXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputh
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8‚‹@‚‹H‚‹PXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputh
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8„€@„€H„€PXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputh
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8√‹@√‹H√‹b-gradient_tape/model/2_BN/FusedBatchNormGradV3hu  »B
È
™void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)™ Ä¿*Ä2Ò8¢¢@¢¢H¢¢PXbmodel/5_conv2d/Conv2Dh
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2ÄM8„@‚ÔHÅÅ	bmodel/1_BN/FusedBatchNormV3hu  »B
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2Ä18¬ò@·’H°‰b-gradient_tape/model/3_BN/FusedBatchNormGradV3hu  »B
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8ÇÕ@ÇÕHÇÕb(gradient_tape/model/1_activation/EluGradhuZUÖB
Á
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8√î@‚ïH·˛Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 512, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8¬Ó@¬ÓH¬Óbmodel/1_BN/FusedBatchNormV3hu  »B
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8„≥@¡˛H·õb-gradient_tape/model/4_BN/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8Éù@·ÅHÅëb>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¬Ü@°ÄH†Éb>gradient_tape/model/batch_normalization_8/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8‚Å@·¯H°Öb>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¢Ó@¡¯HÅ˚b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8‚Ë@°ıHÅ¸b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¬‰@ÅH·˚b>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  »B
¯
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¢‚@‡ÔH°˚b<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8‚‹@ÅÏH°˙b>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  »B
˙
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8‚—@°ÎH‡ıb>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  »B
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿<8‚∑@ÅñH·°bmodel/2_BN/FusedBatchNormV3hu  »B
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8‚î@‚îH‚îb-gradient_tape/model/3_BN/FusedBatchNormGradV3hu  »B
„
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8‚Í@‚ÍH‚Íb<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  »B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2í8Ç¡@Ç¡HÇ¡bmodel/1_DO/dropout/Mul_1huZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2í8¬π@¬πH¬πb&gradient_tape/model/1_DO/dropout/Mul_1huZUÖB
Á
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8·˚@ÄÏH·èXb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8‚‡@‚‡H‚‡b-gradient_tape/model/4_BN/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8°Œ@°ŒH°Œb>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8‚ƒ@‚ƒH‚ƒb>gradient_tape/model/batch_normalization_8/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8¬¡@¬¡H¬¡b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8¬æ@¬æH¬æb>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8‚≥@‚≥H‚≥b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8¢∞@¢∞H¢∞b>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8¬Ø@¬ØH¬Øb>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  »B
Â
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2@8Ç©@Ç©HÇ©b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  »B
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2Ä18Åﬁ@¡ÏH¿Òbmodel/3_BN/FusedBatchNormV3hu  »B
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8·‹@·‹H·‹bmodel/2_BN/FusedBatchNormV3hu  »B
Á
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8·ó@‡ñHÅÅXb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)† Ä¿*Ä2§8¢ì@¢ìH¢ìPXb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputh
§
¬void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`Ä*2†8·È
@·È
H·È
Xb8gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropFilterhu≥™&B
Î
´sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x16_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna4_alignc8_execute_kernel_cudnnv Ä *Ä2…	8°ﬂ
@°ﬂ
H°ﬂ
Xbmodel/1_conv2d/Conv2DhuMUB
•
ƒvoid cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_32x3_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_32x3_nhwc_unity_stride_align4::Params)® Ä¿*Ä2Ò8¢∫
@¢∫
H¢∫
PXb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputh
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8Ç∏
@ÅÍHÅŒb,model/batch_normalization_8/FusedBatchNormV3hu  »B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ç8¢∑
@¢∑
H¢∑
bmodel/2_DO/dropout/Mul_1huZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ç8‚†
@‚†
H‚†
b&gradient_tape/model/2_DO/dropout/Mul_1huZUÖB
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¢ë
@·ˇH¡ëb,model/batch_normalization_4/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¡¯	@°ÙH†Ñb,model/batch_normalization_3/FusedBatchNormV3hu  »B
Ê
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8·Ó	@ÅÛH‡˚b*model/batch_normalization/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8ÇÌ	@¡ÓH¡˛b,model/batch_normalization_5/FusedBatchNormV3hu  »B
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8¿Ï	@ÄÓH¿˛bmodel/4_BN/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8ÇÍ	@·ÂH°Ñb,model/batch_normalization_6/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8·Ë	@†ÔH¡˘b,model/batch_normalization_1/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8·Á	@¡ÍH†˝b,model/batch_normalization_2/FusedBatchNormV3hu  »B
Ë
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2¿*8‚‰	@ÅÈH·˚b,model/batch_normalization_7/FusedBatchNormV3hu  »B
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8Ç‹	@Ç‹	HÇ‹	b(gradient_tape/model/3_activation/EluGradhuZUÖB
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8†…	@ﬂQH¡˜Xbmodel/2_conv2d/Conv2Dhu  »B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2í8¡∞	@¡∞	H¡∞	b$gradient_tape/model/1_DO/dropout/MulhuZUÖB
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2í8¡©	@¡©	H¡©	bmodel/1_DO/dropout/MulhuZUÖB
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8Çû	@Çû	HÇû	bmodel/1_conv2d/BiasAddhu  »B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†≤8Åô	@Åô	HÅô	bmodel/1_activation/Eluhu  »B
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8ÅÛ@ÅÛHÅÛbmodel/3_BN/FusedBatchNormV3hu  »B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8·«@·«H·«bmodel/3_DO/dropout/Mul_1huZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8·∑@·∑H·∑b&gradient_tape/model/3_DO/dropout/Mul_1huZUÖB
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8¡∑@¡∑H¡∑b(gradient_tape/model/4_activation/EluGradhuZUÖB
V
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡´@¡´H¡´bmodel/add_2/addhuZUÖB
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8·ñ@·ñH·ñb,model/batch_normalization_4/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8·Ü@·ÜH·Üb,model/batch_normalization_3/FusedBatchNormV3hu  »B
≤
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8Å˛@Å˛HÅ˛b*model/batch_normalization/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8°¸@°¸H°¸b,model/batch_normalization_6/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8·˚@·˚H·˚b,model/batch_normalization_8/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8°Ù@°ÙH°Ùb,model/batch_normalization_5/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8·Ú@·ÚH·Úb,model/batch_normalization_2/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8ÅÓ@ÅÓHÅÓb,model/batch_normalization_7/FusedBatchNormV3hu  »B
¥
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8‚Î@‚ÎH‚Îb,model/batch_normalization_1/FusedBatchNormV3hu  »B
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2@8¬Î@¬ÎH¬Îbmodel/4_BN/FusedBatchNormV3hu  »B
V
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡€@¡€H¡€bmodel/add_1/addhuZUÖB
Ô
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8·◊@·◊H·◊b)gradient_tape/model/activation_5/ReluGradhuZUÖB
Ô
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8‚÷@‚÷H‚÷b)gradient_tape/model/activation_2/ReluGradhuZUÖB
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8·—@‡FHÅãXbmodel/3_conv2d/Conv2Dhu  »B
Ô
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8Çœ@ÇœHÇœb)gradient_tape/model/activation_1/ReluGradhuZUÖB
Ô
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8Å…@Å…HÅ…b)gradient_tape/model/activation_4/ReluGradhuZUÖB
Ì
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä2H8°«@°«H°«b'gradient_tape/model/activation/ReluGradhuZUÖB
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8°ø@‡;H¡ÉXb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8°æ@°æH°æbmodel/2_conv2d/BiasAddhu  »B
©
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8·º@·ºH·ºbAdam/gradients/AddN_2huZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Åº@ÅºHÅºbmodel/dropout_1/dropout/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Å∫@Å∫HÅ∫b+gradient_tape/model/dropout_5/dropout/Mul_1huZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8‚∑@‚∑H‚∑bmodel/dropout_2/dropout/Mul_1huZUÖB
©
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8°∑@°∑H°∑bAdam/gradients/AddN_4huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8°¥@°¥H°¥b+gradient_tape/model/dropout_1/dropout/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Å¥@Å¥HÅ¥b+gradient_tape/model/dropout_2/dropout/Mul_1huZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡≥@¡≥H¡≥bmodel/dropout/dropout/Mul_1huZUÖB
©
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8¡≥@¡≥H¡≥bAdam/gradients/AddN_3huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·∞@·∞H·∞b+gradient_tape/model/dropout_3/dropout/Mul_1huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·´@·´H·´b)gradient_tape/model/dropout/dropout/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡´@¡´H¡´b+gradient_tape/model/dropout_4/dropout/Mul_1huZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·©@·©H·©b&gradient_tape/model/4_DO/dropout/Mul_1huZUÖB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡©@¡©H¡©bmodel/4_DO/dropout/Mul_1huZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8‡®@‡®H‡®bmodel/dropout_4/dropout/Mul_1huZUÖB
T
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Ç®@Ç®HÇ®bmodel/add/addhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·¶@·¶H·¶bmodel/dropout_3/dropout/Mul_1huZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·•@·•H·•bmodel/dropout_5/dropout/Mul_1huZUÖB
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2‡8¡õ@¿´H¿πb-gradient_tape/model/5_BN/FusedBatchNormGradV3hu  »B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ç8¡ö@¡öH¡öbmodel/2_DO/dropout/MulhuZUÖB
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2 8¡ô@¡ôH¡ôb-gradient_tape/model/5_BN/FusedBatchNormGradV3hu  »B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8·¸@·¸H·¸bmodel/2_activation/Eluhu  »B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ç8°Ò@°ÒH°Òb$gradient_tape/model/2_DO/dropout/MulhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208‡û@‡ûH‡ûb0gradient_tape/model/1_conv2d/BiasAdd/BiasAddGradhuZUÖB
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8°ú@Ä8H°‰Xb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8†ì@†>HÄ’Xbmodel/4_conv2d/Conv2Dhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8°ê@°êH°êbmodel/3_conv2d/BiasAddhu  »B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2†≤8¡ç@¡çH¡çbmodel/1_DO/dropout/Casthu  »B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Äƒ8Åı@ÅıHÅıbmodel/3_activation/Eluhu  »B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÅË@ÅËHÅËb$gradient_tape/model/3_DO/dropout/MulhuZUÖB
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8·Ê@·ÊH·Êbmodel/3_DO/dropout/MulhuZUÖB
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2í8°›@°›H°›bmodel/1_DO/dropout/GreaterEqualhuZUÖB
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8Åƒ@†8H·ãXbmodel/conv2d_2/Conv2Dhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8†∫@†8HÄÇXbmodel/conv2d_3/Conv2Dhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8¿µ@†6H†ˇXb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputhu  »B
¡
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8°µ@Ä8H°˝Xbmodel/conv2d/Conv2Dhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8°¥@Ä5H°ˇXbmodel/conv2d_1/Conv2Dhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8Ä±@¿6H¿˙Xbmodel/5_conv2d/Conv2Dhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8‡¨@†5H¿˜Xbmodel/conv2d_5/Conv2Dhu  »B
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8Ä´@†5H‡ıXbmodel/conv2d_4/Conv2Dhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8°©@°©H°©bmodel/conv2d_3/BiasAddhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8¡¢@¡¢H¡¢bmodel/conv2d_2/BiasAddhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8¡ü@†4H°ÎXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8Åü@Ä4HÅÎXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8·û@Ä4H·ÍXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8Åú@¿7H¡‰Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8‡ò@†4H¿‰Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8·ñ@·ñH·ñbmodel/4_conv2d/BiasAddhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8°î@°îH°îbmodel/conv2d_1/BiasAddhu  »B
å
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8¡í@¡íH¡íbmodel/conv2d/BiasAddhu  »B
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8·ë@·ëH·ëbmodel/conv2d_4/BiasAddhu  »B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Åå@ÅåHÅåbmodel/dropout_2/dropout/MulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8°ã@°ãH°ãbmodel/dropout_1/dropout/MulhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208·â@·âH·âb0gradient_tape/model/2_conv2d/BiasAdd/BiasAddGradhuZUÖB
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8¡â@¡âH¡âbmodel/conv2d_5/BiasAddhu  »B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¿â@¿âH¿âb)gradient_tape/model/dropout_2/dropout/MulhuZUÖB
‘
ˇvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(ê*Ä28¿â@¿âH¿âb-gradient_tape/model/6_BN/FusedBatchNormGradV3hu  »B
È
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8°á@°áH°áb/model/1_DO/dropout/random_uniform/RandomUniformhuZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8°É@°ÉH°Éb)gradient_tape/model/dropout_5/dropout/MulhuZUÖB
‰
£sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnnò Ä *Ä2¿8ÅÉ@ÅÉHÅÉXbmodel/6_conv2d/Conv2Dhu  »A
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¡Ä@¡ÄH¡Äb)gradient_tape/model/dropout_3/dropout/MulhuZUÖB
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8ÄÄ@ÄÄHÄÄbmodel/activation_2/Reluhu  »B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·ˇ@·ˇH·ˇbmodel/4_DO/dropout/MulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¿ˇ@¿ˇH¿ˇbmodel/dropout_3/dropout/MulhuZUÖB
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8·˛@·˛H·˛bmodel/activation_1/Reluhu  »B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Ä˛@Ä˛HÄ˛bmodel/dropout/dropout/MulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·˝@·˝H·˝bmodel/dropout_5/dropout/MulhuZUÖB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8Å˝@Å˝HÅ˝b$gradient_tape/model/4_DO/dropout/MulhuZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8¿¸@¿¸H¿¸b)gradient_tape/model/dropout_1/dropout/MulhuZUÖB
[
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8·˚@·˚H·˚bmodel/activation/Reluhu  »B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·˙@·˙H·˙b)gradient_tape/model/dropout_4/dropout/MulhuZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8°˙@°˙H°˙b'gradient_tape/model/dropout/dropout/MulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2»
8·˘@·˘H·˘bmodel/dropout_4/dropout/MulhuZUÖB
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2‡8†˘@‡¶H¿“bmodel/5_BN/FusedBatchNormV3hu  »B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä©8·¯@·¯H·¯bmodel/4_activation/Eluhu  »B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8Å¯@Å¯HÅ¯bmodel/activation_5/Reluhu  »B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8Ä¯@Ä¯HÄ¯bmodel/activation_3/Reluhu  »B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†*8¡Û@¡ÛH¡Ûbmodel/activation_4/Reluhu  »B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2†8°›@°›H°›bmodel/2_DO/dropout/Casthu  »B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ç8Ä∞@Ä∞HÄ∞bmodel/2_DO/dropout/GreaterEqualhuZUÖB
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2 8¡ù@¡ùH¡ùbmodel/5_BN/FusedBatchNormV3hu  »B
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208·ï@·ïH·ïb0gradient_tape/model/3_conv2d/BiasAdd/BiasAddGradhuZUÖB
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8¿ˇ@¿ˇH¿ˇb(gradient_tape/model/5_activation/EluGradhuZUÖB
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Äƒ8†˚@†˚H†˚bmodel/3_DO/dropout/Casthu  »B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8°Á@°ÁH°Ábmodel/3_DO/dropout/GreaterEqualhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208¡‘@¡‘H¡‘b0gradient_tape/model/4_conv2d/BiasAdd/BiasAddGradhuZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ò8¡…@¡…H¡…b&gradient_tape/model/5_DO/dropout/Mul_1huZUÖB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ò8‡«@‡«H‡«bmodel/5_DO/dropout/Mul_1huZUÖB
ß
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208‡∆@‡∆H‡∆b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZUÖB
È
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2Ä8Ç∆@°íH¡õb-gradient_tape/model/6_BN/FusedBatchNormGradV3hu  »B
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208‡≈@‡≈H‡≈b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208°≈@°≈H°≈b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208°ƒ@°ƒH°ƒb0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZUÖB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) Ä*Ä208‡√@‡√H‡√b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZUÖB
`
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8Å≠@Å≠HÅ≠bmodel/dropout/dropout/Casthu  »B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8¿´@¿´H¿´bmodel/dropout_1/dropout/Casthu  »B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8¿´@¿´H¿´bmodel/dropout_2/dropout/Casthu  »B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8Å™@Å™HÅ™bmodel/dropout_5/dropout/Casthu  »B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8Ä™@Ä™HÄ™bmodel/4_DO/dropout/Casthu  »B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8¡®@¡®H¡®bmodel/dropout_3/dropout/Casthu  »B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä©8·¶@·¶H·¶bmodel/dropout_4/dropout/Casthu  »B
È
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8¿ò@¿òH¿òb/model/3_DO/dropout/random_uniform/RandomUniformhuZUÖB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8‡ç@‡çH‡çb$model/dropout_4/dropout/GreaterEqualhuZUÖB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8†ç@†çH†çb$model/dropout_5/dropout/GreaterEqualhuZUÖB
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8·å@·åH·åbmodel/4_DO/dropout/GreaterEqualhuZUÖB
o
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8‡å@‡åH‡åb"model/dropout/dropout/GreaterEqualhuZUÖB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8¿å@¿åH¿åb$model/dropout_2/dropout/GreaterEqualhuZUÖB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8Äå@ÄåHÄåb$model/dropout_3/dropout/GreaterEqualhuZUÖB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2»
8Åã@ÅãHÅãb$model/dropout_1/dropout/GreaterEqualhuZUÖB
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8ÅÇ@ÅÇHÅÇbmodel/5_conv2d/BiasAddhu  »B
Ó
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8°ﬁ@°ﬁH°ﬁb4model/dropout_2/dropout/random_uniform/RandomUniformhuZUÖB
Ï
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8°›@°›H°›b2model/dropout/dropout/random_uniform/RandomUniformhuZUÖB
È
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8†›@†›H†›b/model/4_DO/dropout/random_uniform/RandomUniformhuZUÖB
√
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8·€@†3H¡®Xbmodel/6_conv2d/Conv2Dhu  »B
Ó
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8¿ÿ@¿ÿH¿ÿb4model/dropout_1/dropout/random_uniform/RandomUniformhuZUÖB
£
‡void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä28Ä◊@Ä◊HÄ◊bmodel/6_BN/FusedBatchNormV3hu  »B
Ó
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8·—@·—H·—b4model/dropout_3/dropout/random_uniform/RandomUniformhuZUÖB
Ó
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8Äœ@ÄœHÄœb4model/dropout_5/dropout/random_uniform/RandomUniformhuZUÖB
Ó
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8†∆@†∆H†∆b4model/dropout_4/dropout/random_uniform/RandomUniformhuZUÖB
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8‡ø@Ä2H‡çXb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputhu  »B
◊
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) Ä!*Ä2Ä8·Ω@ÄúH·°bmodel/6_BN/FusedBatchNormV3hu  »B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ò8°Ω@°ΩH°Ωbmodel/5_DO/dropout/MulhuZUÖB
ƒ
Üvoid cutlass::Kernel<cutlass_80_tensorop_s1688gemm_256x64_16x4_tt_align1>(cutlass_80_tensorop_s1688gemm_256x64_16x4_tt_align1::Params) ÄÄ*Ä2$8Äª@ÄªHÄªPXbmodel/conv2d_6/Conv2Dh
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ò8†µ@†µH†µb$gradient_tape/model/5_DO/dropout/MulhuZUÖB
[
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2êN8¿™@¿™H¿™bmodel/5_activation/Eluhu  »B
Û
ïvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 2)8‡ü@‡üH‡üXb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
Á
Üvoid cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x256_16x4_tn_align1>(cutlass_80_tensorop_s1688gemm_64x256_16x4_tn_align1::Params)Ù ÄÄ*Ä2†8·ú@·úH·úPXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputh
í
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2H8¿ı@¿ıH¿ıb(gradient_tape/model/6_activation/EluGradhuZUÖB
˝
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)Ä!*  2 8†›@†›H†›b0gradient_tape/model/5_conv2d/BiasAdd/BiasAddGradhuZUÖB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8°◊@°◊H°◊bmodel/6_DO/dropout/Mul_1huZUÖB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8†’@†’H†’b&gradient_tape/model/6_DO/dropout/Mul_1huZUÖB
\
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2êN8¿—@¿—H¿—bmodel/5_DO/dropout/Casthu  »B
⁄
övoid cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)!Ä!*Ä2» 8¡¿@¡¿H¡¿Xbmodel/1_conv2d/Conv2Dhu  »B
Â
ávoid cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*Ä2Ä 8†∑@¿2H‡ÑXb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputhu  »B
Û
ïvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 28ÅÆ@ÅÆHÅÆXb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
È
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8°´@°´H°´b/model/5_DO/dropout/random_uniform/RandomUniformhuZUÖB
Ç
ampere_sgemm_32x32_sliced1x4_ntVÄÄ*Ä2ÿ8¿™@¿™H¿™Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhu≥™&B
˛
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä208¡°@¡°H¡°b$Adam/Adam/update_4/ResourceApplyAdamhuZUÖB
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äõ@ÄõHÄõbmodel/6_DO/dropout/MulhuZUÖB
é
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8Äô@ÄôHÄôbmodel/6_conv2d/BiasAddhu  »B
[
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä$8†î@†îH†îbmodel/6_activation/Eluhu  »B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8†å@†åH†åb$gradient_tape/model/6_DO/dropout/MulhuZUÖB
Ü
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)Ä!*  2 8†t@†tH†tb0gradient_tape/model/6_conv2d/BiasAdd/BiasAddGradhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä208†n@†nH†nb$Adam/Adam/update_8/ResourceApplyAdamhuZUÖB
Ê
ïvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*Ä2H8¿d@¿dH¿db/model/6_DO/dropout/random_uniform/RandomUniformhuZUÖB
Y
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*Ä2Ä$8‡_@‡_H‡_bmodel/6_DO/dropout/Casthu  »B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8ÄT@ÄTHÄTbmodel/6_DO/dropout/GreaterEqualhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208†E@†EH†EXbmodel/2_conv2d/Conv2DhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208¿D@¿DH¿DXb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä208‡?@‡?H‡?b%Adam/Adam/update_12/ResourceApplyAdamhuZUÖB

ïvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 28Ä=@Ä=HÄ=Xb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhu  »B
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208†7@†7H†7Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208†6@†6H†6Xbmodel/3_conv2d/Conv2DhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208¿4@¿4H¿4Xb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
†
≈void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*) * 28¿.@¿.H¿.Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhu  »B
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208¿)@¿)H¿)Xb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208‡(@‡(H‡(Xbmodel/4_conv2d/Conv2DhuZUÖB
’
Ñvoid cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1::Params)v ÄÄ*Ä28‡'@‡'H‡'PXb'gradient_tape/model/dense/MatMul/MatMulhugUÖA
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208Ä'@Ä'HÄ'Xb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28‡$@‡$H‡$bAdam/addhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8¿$@¿$H¿$b%Adam/Adam/update_20/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8†$@†$H†$b%Adam/Adam/update_36/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8¿#@¿#H¿#b%Adam/Adam/update_16/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8¿#@¿#H¿#b%Adam/Adam/update_30/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8†#@†#H†#b%Adam/Adam/update_26/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2$8‡"@‡"H‡"b%Adam/Adam/update_40/ResourceApplyAdamhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä208¿"@¿"H¿"Xb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
Ç
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*Ä28†!@†!H†!b5gradient_tape/binary_focal_crossentropy/DynamicStitchhuZUÖB
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡ @‡ H‡ bAdam/PowhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿ @¿ H¿ b$Adam/Adam/update_6/ResourceApplyAdamhuZUÖB
Ü
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)Ä!*  2 8† @† H† b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_56/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_46/ResourceApplyAdamhuZUÖB
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*Ä28‡@‡H‡bAdam/Cast_1hu  »B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†b
Adam/Pow_1huZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_38/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_18/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_10/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_22/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_34/ResourceApplyAdamhuZUÖB
§
Îvoid dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > >(cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >) Ä*Ä2 8†@†H†Xbmodel/dense/MatMulhu  »B
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_14/ResourceApplyAdamhuZUÖB
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8†@†H†Xb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8†@†H†Xbmodel/conv2d_3/Conv2DhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb$Adam/Adam/update_2/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_48/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_28/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_45/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_52/ResourceApplyAdamhuZUÖB
˘
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b"Adam/Adam/update/ResourceApplyAdamhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8¿@¿H¿Xbmodel/conv2d_2/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28°@°H°b%Adam/Adam/update_32/ResourceApplyAdamhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8°@°H°Xbmodel/conv2d_4/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_24/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_42/ResourceApplyAdamhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b$Adam/Adam/update_5/ResourceApplyAdamhuZUÖB
º
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8†@†H†Xbmodel/conv2d/Conv2DhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8†@†H†Xbmodel/conv2d_1/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_44/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_50/ResourceApplyAdamhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28¡@¡H¡Xbmodel/1_conv2d/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_31/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_54/ResourceApplyAdamhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Å@ÅHÅXb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXbmodel/conv2d_5/Conv2DhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8‡@‡H‡Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_15/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_43/ResourceApplyAdamhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28¿@¿H¿Xb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
É
µstd::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, false, false, 7, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)• Ä* 2H8Ä@ÄHÄb)gradient_tape/model/dense/MatMul/MatMul_1hu  »A
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_33/ResourceApplyAdamhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28‡@‡H‡Xbmodel/5_conv2d/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_39/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_49/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_37/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb%Adam/Adam/update_55/ResourceApplyAdamhuZUÖB
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_25/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_35/ResourceApplyAdamhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b$Adam/Adam/update_7/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¡@¡H¡b%Adam/Adam/update_29/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_19/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_23/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_53/ResourceApplyAdamhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8¿@¿H¿Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b$Adam/Adam/update_3/ResourceApplyAdamhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8ü@üHüXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZUÖB
÷
övoid cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)!Ä!*Ä2@8Ä@ÄHÄXbmodel/1_conv2d/Conv2Dhu  »B
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28Ä@ÄHÄXb8gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä2$8Ä@ÄHÄXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_13/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_17/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_21/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_27/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_41/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b%Adam/Adam/update_47/ResourceApplyAdamhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28‡@‡H‡b$Adam/Adam/update_9/ResourceApplyAdamhuZUÖB
æ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28‡@‡H‡Xbmodel/6_conv2d/Conv2DhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28¿@¿H¿b%Adam/Adam/update_51/ResourceApplyAdamhuZUÖB
˚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b$Adam/Adam/update_1/ResourceApplyAdamhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28†@†H†b%Adam/Adam/update_11/ResourceApplyAdamhuZUÖB
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28†@†H†Xb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputhuZUÖB
¸
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28ø@øHøb%Adam/Adam/update_57/ResourceApplyAdamhuZUÖB
É
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)Ä!* 28†@†H†b-gradient_tape/model/dense/BiasAdd/BiasAddGradhuMUB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28Ä@ÄHÄXb8gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
·
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*Ä28†@†H†Xb8gradient_tape/model/1_conv2d/Conv2D/Conv2DBackpropFilterhuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28ü@üHübAssignAddVariableOp_2huZUÖB
†
Ávoid reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(float const*, float, cublasGemvTensorStridedBatched<float>, int, float const*, float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, cublasPointerMode_t, cublasLtEpilogue_t, cublasGemvTensorStridedBatched<biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type const>)Ä*Ä2 8Ä@ÄHÄXbmodel/dense/MatMulhu  »B
n
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*Ä28‡@‡H‡b,gradient_tape/binary_focal_crossentropy/Casthu  »B
ã
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä2`8‡@‡H‡bmodel/conv2d_6/BiasAddhu  »B
Á
ñvoid tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2 8‡@‡H‡b0gradient_tape/model/5_conv2d/BiasAdd/BiasAddGradhuZUÖB
|
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b4gradient_tape/binary_focal_crossentropy/Reciprocal_1hu  »B
·
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28†@†H†b<gradient_tape/binary_focal_crossentropy/weighted_loss/Tile_1huZUÖB
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*Ä28Ä@ÄHÄbCasthu  »B
p
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b/gradient_tape/binary_focal_crossentropy/truedivhuZUÖB
ò
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28†@†H†bAssignAddVariableOphuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28†@†H†bAssignAddVariableOp_1huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_3huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_1huZUÖB
Ë
ñvoid tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*Ä28Ä@ÄHÄb0gradient_tape/model/6_conv2d/BiasAdd/BiasAddGradhu  »B
§
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28·@·H·bAdam/gradients/AddNhuZUÖB
Ö
%LessEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28¿@¿H¿b?gradient_tape/binary_focal_crossentropy/clip_by_value/LessEqualhuZUÖB
z
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b2gradient_tape/binary_focal_crossentropy/Reciprocalhu  »B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†bbinary_focal_crossentropy/mulhuZUÖB
å
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbFgradient_tape/binary_focal_crossentropy/weighted_loss/value/div_no_nanhuZUÖB
ã
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbBgradient_tape/binary_focal_crossentropy/clip_by_value/GreaterEqualhuZUÖB
D
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*Ä28Ä@ÄHÄhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/binary_focal_crossentropy/Pow/mulhuZUÖB
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbbinary_focal_crossentropy/add_2huZUÖB
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbbinary_focal_crossentropy/sub_1huZUÖB
ò
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/Adam/AssignAddVariableOphuZUÖB
ê
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_4huZUÖB
à
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*Ä28Ä@ÄHÄbmodel/dense/BiasAddhu  »B
Î
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbSum_2hu  »B
^
 Log_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28ˇ@ˇHˇbbinary_focal_crossentropy/Loghu  »B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28ˇ@ˇHˇbbinary_focal_crossentropy/subhuZUÖB
t
$Minimum_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b/binary_focal_crossentropy/clip_by_value/MinimumhuZUÖB
Ñ
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b>gradient_tape/binary_focal_crossentropy/clip_by_value/SelectV2hu  »B
Ë
ñvoid tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*Ä28‡@‡H‡b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhu  »B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿bbinary_focal_crossentropy/add_1huZUÖB
s
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b-binary_focal_crossentropy/weighted_loss/valuehuZUÖB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿bMulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿bbinary_focal_crossentropy/mul_1huZUÖB
Â
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*Ä28¿@¿H¿b"gradient_tape/model/dense/ReluGradhuZUÖB
l
$Maximum_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†b'binary_focal_crossentropy/clip_by_valuehuZUÖB
a
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†bbinary_focal_crossentropy/PowhuZUÖB
`
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†bSquaredDifferencehuZUÖB
ë
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28†@†H†b+binary_focal_crossentropy/weighted_loss/Sumhu  »B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*Ä28Å@ÅHÅbCast_2hu  »B
P
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb
div_no_nanhuZUÖB
`
 Log_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbbinary_focal_crossentropy/Log_1hu  »B
^
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbbinary_focal_crossentropy/Neghu  »B
Ü
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/binary_focal_crossentropy/clip_by_value/SelectV2_1hu  »B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28ˇ@ˇHˇbbinary_focal_crossentropy/addhuZUÖB
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28·@·H·bbinary_focal_crossentropy/sub_2huZUÖB
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡bbinary_focal_crossentropy/add_3huZUÖB
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*Ä28‡@‡H‡b9binary_focal_crossentropy/weighted_loss/num_elements/Casthu  »B
R
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡bdiv_no_nan_1huZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡bbinary_focal_crossentropy/mul_2huZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡bbinary_focal_crossentropy/mul_4huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b1gradient_tape/binary_focal_crossentropy/Pow/mul_1huZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b+gradient_tape/binary_focal_crossentropy/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b/gradient_tape/binary_focal_crossentropy/mul/MulhuZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b-gradient_tape/binary_focal_crossentropy/mul_1huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b1gradient_tape/binary_focal_crossentropy/mul_1/MulhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡b1gradient_tape/binary_focal_crossentropy/mul_4/MulhuZUÖB
R
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28‡@‡H‡bmodel/dense/Reluhu  »B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿bbinary_focal_crossentropy/mul_3huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b1gradient_tape/binary_focal_crossentropy/mul_2/MulhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b1gradient_tape/binary_focal_crossentropy/mul_3/MulhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28¿@¿H¿b3gradient_tape/binary_focal_crossentropy/mul_4/Mul_1huZUÖB
`
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*Ä28†@†H†bbinary_focal_crossentropy/Casthu  »B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†b1gradient_tape/binary_focal_crossentropy/sub_2/Neghu  »B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†b1gradient_tape/binary_focal_crossentropy/sub_5/Neghu  »B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28ü@üHüb1gradient_tape/binary_focal_crossentropy/sub_1/Neghu  »B
l
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb+gradient_tape/binary_focal_crossentropy/Neghu  »B
á
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28†@†H†b@gradient_tape/binary_focal_crossentropy/clip_by_value/zeros_likehuZUÖB
â
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/binary_focal_crossentropy/clip_by_value/zeros_like_1huZUÖB