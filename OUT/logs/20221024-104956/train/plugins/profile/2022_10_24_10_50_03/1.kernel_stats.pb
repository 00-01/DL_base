
�
�sm80_xmma_dgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn� ��*�2�	8���@���H���Xb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�28���@���H���Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)� ��*�2�8碉@碉H碉PXbmodel/2_conv2d/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�28���@���H���Xb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8���@���H���PXb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)� ��*�2�8���@���H���PXbmodel/3_conv2d/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi128ELi32EEENS4_52Conv2dWgradOutputGradientTileAccessIteratorOptimizedINS_11MatrixShapeILi64ELi32EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi64ELi32EEELi128ENSF_ILi8ELi4EEELi4EEENS_12AlignedArrayISC_Li4ELi16EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NS_6layout40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESI_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_48Conv2dWgradActivationTileAccessIteratorOptimizedINSA_ILi32ELi128EEESC_NSE_INSF_ILi128ELi32EEELi128ESH_Li4EEESK_EENSN_ISW_SC_NSO_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESY_Li16EEELSU_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi64ELi32EEESC_SQ_SC_S11_fNSO_8RowMajorENS15_17MmaTensorOpPolicyINSS_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S18_SC_NSO_11ColumnMajorEfS18_NSS_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1I_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1H_Li1ENS1M_22PredicatedTileIteratorINS1M_26OutputTileOptimalThreadMapINS1M_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfLb0EEENS1L_4warp24FragmentIteratorTensorOpIS17_S1B_fNS_5ArrayIfLi4ELb1EEES18_EENS1V_20TileIteratorTensorOpIS17_S1B_fS18_EENS1M_18SharedLoadIteratorINS1T_18CompactedThreadMapEfLi16EEENS1L_6thread17LinearCombinationIfLi4EffLNS25_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEELi2ELi1EEENS13_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2	8��i@��iH��iXb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��V@��VH��Vb(gradient_tape/model/2_activation/EluGradhuZU�B
�
�void wgrad_alg0_engine_NHWC<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)N�*2�8�E@�EH�EXb8gradient_tape/model/1_conv2d/Conv2D/Conv2DBackpropFilterhu  HB
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8�9@�9H�9PXb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4::Params)� ��*�2�8��2@��2H��2PXbmodel/4_conv2d/Conv2Dh
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��+@��+H��+Xb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhu��&B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��+@��+H��+Xb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhu��&B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8Ʀ+@Ʀ+HƦ+Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu��&B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��+@��+H��+Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu��&B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��*@��*H��*Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhu��&B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��*@��*H��*Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhu��&B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8Ř'@�4H��&Xb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8�%@�%H�%b/model/2_DO/dropout/random_uniform/RandomUniformhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�M8ĺ@��H��b-gradient_tape/model/1_BN/FusedBatchNormGradV3hu  �B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bmodel/5_DO/dropout/GreaterEqualhuZU�B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b-gradient_tape/model/1_BN/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�<8��@��H��b-gradient_tape/model/2_BN/FusedBatchNormGradV3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b)gradient_tape/model/activation_3/ReluGradhuZU�B
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2	�8��@��H��Xb8gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropFilterhu��&B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/conv2d/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8�@�H�PXbmodel/conv2d_5/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/conv2d_4/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/conv2d_2/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/conv2d_3/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/conv2d_1/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputh
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b-gradient_tape/model/2_BN/FusedBatchNormGradV3hu  �B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4::Params)� ��*�2�8��@��H��PXbmodel/5_conv2d/Conv2Dh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�M8��@��H��	bmodel/1_BN/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�18@��H��b-gradient_tape/model/3_BN/FusedBatchNormGradV3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b(gradient_tape/model/1_activation/EluGradhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8Ô@�H��Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 512, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��bmodel/1_BN/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8�@��H�b-gradient_tape/model/4_BN/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@�H��b>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8@��H��b>gradient_tape/model/batch_normalization_8/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8�@��H��b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��@��H��b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�<8�@��H�bmodel/2_BN/FusedBatchNormV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8�@�H�b-gradient_tape/model/3_BN/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  �B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/1_DO/dropout/Mul_1huZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8¹@¹H¹b&gradient_tape/model/1_DO/dropout/Mul_1huZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@��H�Xb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b-gradient_tape/model/4_BN/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b>gradient_tape/model/batch_normalization_8/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8¾@¾H¾b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8�@�H�b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8¯@¯H¯b>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2@8��@��H��b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�18��@��H��bmodel/3_BN/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��bmodel/2_BN/FusedBatchNormV3hu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8�@��H��Xb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_nhwc_unity_stride_align4::Params)� ��*�2�8��@��H��PXb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputh
�
�void wgrad_alg0_engine_NHWC<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)`�*2�8��
@��
H��
Xb8gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropFilterhu��&B
�
�sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x16_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna4_alignc8_execute_kernel_cudnnv � *�2�	8��
@��
H��
Xbmodel/1_conv2d/Conv2DhuMUB
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_32x3_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_32x3_nhwc_unity_stride_align4::Params)� ��*�2�8��
@��
H��
PXb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��
@��H��b,model/batch_normalization_8/FusedBatchNormV3hu  �B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��
@��
H��
bmodel/2_DO/dropout/Mul_1huZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�
@�
H�
b&gradient_tape/model/2_DO/dropout/Mul_1huZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��
@��H��b,model/batch_normalization_4/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_3/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b*model/batch_normalization/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_5/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��bmodel/4_BN/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_6/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_1/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_2/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�*8��	@��H��b,model/batch_normalization_7/FusedBatchNormV3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��	@��	H��	b(gradient_tape/model/3_activation/EluGradhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��	@�QH��Xbmodel/2_conv2d/Conv2Dhu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��	@��	H��	b$gradient_tape/model/1_DO/dropout/MulhuZU�B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��	@��	H��	bmodel/1_DO/dropout/MulhuZU�B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��	@��	H��	bmodel/1_conv2d/BiasAddhu  �B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8��	@��	H��	bmodel/1_activation/Eluhu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��bmodel/3_BN/FusedBatchNormV3hu  �B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/3_DO/dropout/Mul_1huZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�@�H�b&gradient_tape/model/3_DO/dropout/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b(gradient_tape/model/4_activation/EluGradhuZU�B
V
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/add_2/addhuZU�B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8�@�H�b,model/batch_normalization_4/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8�@�H�b,model/batch_normalization_3/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b*model/batch_normalization/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_6/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_8/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_5/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_2/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_7/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��b,model/batch_normalization_1/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2@8��@��H��bmodel/4_BN/FusedBatchNormV3hu  �B
V
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/add_1/addhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b)gradient_tape/model/activation_5/ReluGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b)gradient_tape/model/activation_2/ReluGradhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�FH��Xbmodel/3_conv2d/Conv2Dhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b)gradient_tape/model/activation_1/ReluGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b)gradient_tape/model/activation_4/ReluGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b'gradient_tape/model/activation/ReluGradhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�;H��Xb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/2_conv2d/BiasAddhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8�@�H�bAdam/gradients/AddN_2huZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_1/dropout/Mul_1huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b+gradient_tape/model/dropout_5/dropout/Mul_1huZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�bmodel/dropout_2/dropout/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��bAdam/gradients/AddN_4huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b+gradient_tape/model/dropout_1/dropout/Mul_1huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b+gradient_tape/model/dropout_2/dropout/Mul_1huZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout/dropout/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��bAdam/gradients/AddN_3huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�b+gradient_tape/model/dropout_3/dropout/Mul_1huZU�B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�b)gradient_tape/model/dropout/dropout/Mul_1huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b+gradient_tape/model/dropout_4/dropout/Mul_1huZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�b&gradient_tape/model/4_DO/dropout/Mul_1huZU�B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/4_DO/dropout/Mul_1huZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�bmodel/dropout_4/dropout/Mul_1huZU�B
T
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/add/addhuZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�bmodel/dropout_3/dropout/Mul_1huZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8�@�H�bmodel/dropout_5/dropout/Mul_1huZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�8��@��H��b-gradient_tape/model/5_BN/FusedBatchNormGradV3hu  �B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/2_DO/dropout/MulhuZU�B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�2 8��@��H��b-gradient_tape/model/5_BN/FusedBatchNormGradV3hu  �B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8��@��H��bmodel/2_activation/Eluhu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b$gradient_tape/model/2_DO/dropout/MulhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/1_conv2d/BiasAdd/BiasAddGradhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�8H��Xb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�>H��Xbmodel/4_conv2d/Conv2Dhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/3_conv2d/BiasAddhu  �B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/1_DO/dropout/Casthu  �B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8��@��H��bmodel/3_activation/Eluhu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b$gradient_tape/model/3_DO/dropout/MulhuZU�B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/3_DO/dropout/MulhuZU�B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bmodel/1_DO/dropout/GreaterEqualhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�8H�Xbmodel/conv2d_2/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�8H��Xbmodel/conv2d_3/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�6H��Xb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�8H��Xbmodel/conv2d/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�5H��Xbmodel/conv2d_1/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�6H��Xbmodel/5_conv2d/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8�@�5H��Xbmodel/conv2d_5/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�5H��Xbmodel/conv2d_4/Conv2Dhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/conv2d_3/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/conv2d_2/BiasAddhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�4H��Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�4H��Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8�@�4H��Xb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�7H��Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�4H��Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8�@�H�bmodel/4_conv2d/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/conv2d_1/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/conv2d/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8�@�H�bmodel/conv2d_4/BiasAddhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_2/dropout/MulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_1/dropout/MulhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208�@�H�b0gradient_tape/model/2_conv2d/BiasAdd/BiasAddGradhuZU�B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/conv2d_5/BiasAddhu  �B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b)gradient_tape/model/dropout_2/dropout/MulhuZU�B
�
�void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(�*�28��@��H��b-gradient_tape/model/6_BN/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b/model/1_DO/dropout/random_uniform/RandomUniformhuZU�B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b)gradient_tape/model/dropout_5/dropout/MulhuZU�B
�
�sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn� � *�2�8��@��H��Xbmodel/6_conv2d/Conv2Dhu  �A
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b)gradient_tape/model/dropout_3/dropout/MulhuZU�B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation_2/Reluhu  �B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/4_DO/dropout/MulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_3/dropout/MulhuZU�B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation_1/Reluhu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout/dropout/MulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_5/dropout/MulhuZU�B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b$gradient_tape/model/4_DO/dropout/MulhuZU�B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b)gradient_tape/model/dropout_1/dropout/MulhuZU�B
[
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation/Reluhu  �B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b)gradient_tape/model/dropout_4/dropout/MulhuZU�B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��b'gradient_tape/model/dropout/dropout/MulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�
8��@��H��bmodel/dropout_4/dropout/MulhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�8��@�H��bmodel/5_BN/FusedBatchNormV3hu  �B
\
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8��@��H��bmodel/4_activation/Eluhu  �B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation_5/Reluhu  �B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation_3/Reluhu  �B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�*8��@��H��bmodel/activation_4/Reluhu  �B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/2_DO/dropout/Casthu  �B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bmodel/2_DO/dropout/GreaterEqualhuZU�B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2 8��@��H��bmodel/5_BN/FusedBatchNormV3hu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208�@�H�b0gradient_tape/model/3_conv2d/BiasAdd/BiasAddGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b(gradient_tape/model/5_activation/EluGradhuZU�B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/3_DO/dropout/Casthu  �B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bmodel/3_DO/dropout/GreaterEqualhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/4_conv2d/BiasAdd/BiasAddGradhuZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b&gradient_tape/model/5_DO/dropout/Mul_1huZU�B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/5_DO/dropout/Mul_1huZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�8��@��H��b-gradient_tape/model/6_BN/FusedBatchNormGradV3hu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZU�B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�208��@��H��b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZU�B
`
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/dropout/dropout/Casthu  �B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/dropout_1/dropout/Casthu  �B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/dropout_2/dropout/Casthu  �B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/dropout_5/dropout/Casthu  �B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/4_DO/dropout/Casthu  �B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8��@��H��bmodel/dropout_3/dropout/Casthu  �B
b
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2��8�@�H�bmodel/dropout_4/dropout/Casthu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b/model/3_DO/dropout/random_uniform/RandomUniformhuZU�B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b$model/dropout_4/dropout/GreaterEqualhuZU�B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b$model/dropout_5/dropout/GreaterEqualhuZU�B
l
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8�@�H�bmodel/4_DO/dropout/GreaterEqualhuZU�B
o
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b"model/dropout/dropout/GreaterEqualhuZU�B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b$model/dropout_2/dropout/GreaterEqualhuZU�B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b$model/dropout_3/dropout/GreaterEqualhuZU�B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8��@��H��b$model/dropout_1/dropout/GreaterEqualhuZU�B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/5_conv2d/BiasAddhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b4model/dropout_2/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b2model/dropout/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b/model/4_DO/dropout/random_uniform/RandomUniformhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�3H��Xbmodel/6_conv2d/Conv2Dhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b4model/dropout_1/dropout/random_uniform/RandomUniformhuZU�B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, int, 128, true, 1, true>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�28��@��H��bmodel/6_BN/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b4model/dropout_3/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b4model/dropout_5/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b4model/dropout_4/dropout/random_uniform/RandomUniformhuZU�B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8�@�2H��Xb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*) �!*�2�8�@��H�bmodel/6_BN/FusedBatchNormV3hu  �B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/5_DO/dropout/MulhuZU�B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_256x64_16x4_tt_align1>(cutlass_80_tensorop_s1688gemm_256x64_16x4_tt_align1::Params)� ��*�2$8��@��H��PXbmodel/conv2d_6/Conv2Dh
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b$gradient_tape/model/5_DO/dropout/MulhuZU�B
[
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�N8��@��H��bmodel/5_activation/Eluhu  �B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 2)8��@��H��Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x256_16x4_tn_align1>(cutlass_80_tensorop_s1688gemm_64x256_16x4_tn_align1::Params)� ��*�2�8�@�H�PXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2H8��@��H��b(gradient_tape/model/6_activation/EluGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!*  2 8��@��H��b0gradient_tape/model/5_conv2d/BiasAdd/BiasAddGradhuZU�B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/6_DO/dropout/Mul_1huZU�B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b&gradient_tape/model/6_DO/dropout/Mul_1huZU�B
\
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�N8��@��H��bmodel/5_DO/dropout/Casthu  �B
�
�void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)!�!*�2� 8��@��H��Xbmodel/1_conv2d/Conv2Dhu  �B
�
�void cudnn::ops::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, unsigned long)*�2� 8��@�2H��Xb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputhu  �B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 28��@��H��Xb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8��@��H��b/model/5_DO/dropout/random_uniform/RandomUniformhuZU�B
�
ampere_sgemm_32x32_sliced1x4_ntV��*�2�8��@��H��Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhu��&B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�208��@��H��b$Adam/Adam/update_4/ResourceApplyAdamhuZU�B
[
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmodel/6_DO/dropout/MulhuZU�B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8��@��H��bmodel/6_conv2d/BiasAddhu  �B
[
 Elu_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�$8��@��H��bmodel/6_activation/Eluhu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b$gradient_tape/model/6_DO/dropout/MulhuZU�B
�
�void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!*  2 8�t@�tH�tb0gradient_tape/model/6_conv2d/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�208�n@�nH�nb$Adam/Adam/update_8/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2H8�d@�dH�db/model/6_DO/dropout/random_uniform/RandomUniformhuZU�B
Y
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�$8�_@�_H�_bmodel/6_DO/dropout/Casthu  �B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8�T@�TH�Tbmodel/6_DO/dropout/GreaterEqualhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�E@�EH�EXbmodel/2_conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�D@�DH�DXb7gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�208�?@�?H�?b%Adam/Adam/update_12/ResourceApplyAdamhuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 128>, cutlass_cudnn::epilogue::thread::LinearCombination<float, 4, float, float, (cutlass_cudnn::epilogue::thread::ScaleType::Kind)0, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 4>, 4>::Params)$* 28�=@�=H�=Xb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�7@�7H�7Xb8gradient_tape/model/2_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�6@�6H�6Xbmodel/3_conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�4@�4H�4Xb7gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropInputhuZU�B
�
�void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*) * 28�.@�.H�.Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�)@�)H�)Xb8gradient_tape/model/3_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�(@�(H�(Xbmodel/4_conv2d/Conv2DhuZU�B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1::Params)v ��*�28�'@�'H�'PXb'gradient_tape/model/dense/MatMul/MatMulhugU�A
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�'@�'H�'Xb7gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropInputhuZU�B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�$@�$H�$bAdam/addhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�$@�$H�$b%Adam/Adam/update_20/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�$@�$H�$b%Adam/Adam/update_36/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�#@�#H�#b%Adam/Adam/update_16/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�#@�#H�#b%Adam/Adam/update_30/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�#@�#H�#b%Adam/Adam/update_26/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2$8�"@�"H�"b%Adam/Adam/update_40/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�208�"@�"H�"Xb8gradient_tape/model/4_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*�28�!@�!H�!b5gradient_tape/binary_focal_crossentropy/DynamicStitchhuZU�B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28� @� H� bAdam/PowhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b$Adam/Adam/update_6/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!*  2 8� @� H� b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_56/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_46/ResourceApplyAdamhuZU�B
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�@�H�bAdam/Cast_1hu  �B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b
Adam/Pow_1huZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_38/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_18/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_10/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_22/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_34/ResourceApplyAdamhuZU�B
�
�void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > >(cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >) �*�2 8�@�H�Xbmodel/dense/MatMulhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_14/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb5gradient_tape/model/conv2d/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d_3/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_2/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_48/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_28/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_45/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_52/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b"Adam/Adam/update/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d_2/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_32/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d_4/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_24/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_42/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_5/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d_1/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_44/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_50/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xbmodel/1_conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_31/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_54/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xbmodel/conv2d_5/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_15/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_43/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xb7gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropInputhuZU�B
�
�std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, false, false, 7, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)� �* 2H8�@�H�b)gradient_tape/model/dense/MatMul/MatMul_1hu  �A
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_33/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xbmodel/5_conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_39/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_49/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_37/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_55/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_25/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_35/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_7/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_29/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_19/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_23/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_53/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_3/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)!�!*�2@8�@�H�Xbmodel/1_conv2d/Conv2Dhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xb8gradient_tape/model/5_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�2$8�@�H�Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_13/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_17/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_21/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_27/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_41/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_47/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_9/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xbmodel/6_conv2d/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_51/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b$Adam/Adam/update_1/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_11/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 1, 2, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xb7gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b%Adam/Adam/update_57/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!* 28�@�H�b-gradient_tape/model/dense/BiasAdd/BiasAddGradhuMUB
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xb8gradient_tape/model/6_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 0, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*�28�@�H�Xb8gradient_tape/model/1_conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_2huZU�B
�
�void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(float const*, float, cublasGemvTensorStridedBatched<float>, int, float const*, float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, cublasPointerMode_t, cublasLtEpilogue_t, cublasGemvTensorStridedBatched<biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type const>)�*�2 8�@�H�Xbmodel/dense/MatMulhu  �B
n
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�b,gradient_tape/binary_focal_crossentropy/Casthu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�2`8�@�H�bmodel/conv2d_6/BiasAddhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2 8�@�H�b0gradient_tape/model/5_conv2d/BiasAdd/BiasAddGradhuZU�B
|
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4gradient_tape/binary_focal_crossentropy/Reciprocal_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�@�H�b<gradient_tape/binary_focal_crossentropy/weighted_loss/Tile_1huZU�B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�bCasthu  �B
p
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b/gradient_tape/binary_focal_crossentropy/truedivhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOphuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_3huZU�B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_1huZU�B
�
�void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*�28�@�H�b0gradient_tape/model/6_conv2d/BiasAdd/BiasAddGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddNhuZU�B
�
%LessEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b?gradient_tape/binary_focal_crossentropy/clip_by_value/LessEqualhuZU�B
z
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b2gradient_tape/binary_focal_crossentropy/Reciprocalhu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/mulhuZU�B
�
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bFgradient_tape/binary_focal_crossentropy/weighted_loss/value/div_no_nanhuZU�B
�
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bBgradient_tape/binary_focal_crossentropy/clip_by_value/GreaterEqualhuZU�B
D
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�@�H�huZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b/gradient_tape/binary_focal_crossentropy/Pow/mulhuZU�B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/add_2huZU�B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/sub_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/Adam/AssignAddVariableOphuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_4huZU�B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�@�H�bmodel/dense/BiasAddhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�@�H�bSum_2hu  �B
^
 Log_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/Loghu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/subhuZU�B
t
$Minimum_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b/binary_focal_crossentropy/clip_by_value/MinimumhuZU�B
�
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b>gradient_tape/binary_focal_crossentropy/clip_by_value/SelectV2hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*�28�@�H�b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/add_1huZU�B
s
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b-binary_focal_crossentropy/weighted_loss/valuehuZU�B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bMulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b"gradient_tape/model/dense/ReluGradhuZU�B
l
$Maximum_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b'binary_focal_crossentropy/clip_by_valuehuZU�B
a
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/PowhuZU�B
`
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bSquaredDifferencehuZU�B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�@�H�b+binary_focal_crossentropy/weighted_loss/Sumhu  �B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�bCast_2hu  �B
P
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b
div_no_nanhuZU�B
`
 Log_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/Log_1hu  �B
^
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/Neghu  �B
�
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/binary_focal_crossentropy/clip_by_value/SelectV2_1hu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/addhuZU�B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/sub_2huZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/add_3huZU�B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�b9binary_focal_crossentropy/weighted_loss/num_elements/Casthu  �B
R
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bdiv_no_nan_1huZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/mul_2huZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/mul_4huZU�B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/Pow/mul_1huZU�B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b+gradient_tape/binary_focal_crossentropy/mulhuZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b/gradient_tape/binary_focal_crossentropy/mul/MulhuZU�B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b-gradient_tape/binary_focal_crossentropy/mul_1huZU�B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/mul_1/MulhuZU�B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/mul_4/MulhuZU�B
R
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmodel/dense/Reluhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/mul_3huZU�B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/mul_2/MulhuZU�B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/mul_3/MulhuZU�B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b3gradient_tape/binary_focal_crossentropy/mul_4/Mul_1huZU�B
`
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�@�H�bbinary_focal_crossentropy/Casthu  �B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/sub_2/Neghu  �B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/sub_5/Neghu  �B
r
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b1gradient_tape/binary_focal_crossentropy/sub_1/Neghu  �B
l
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b+gradient_tape/binary_focal_crossentropy/Neghu  �B
�
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/binary_focal_crossentropy/clip_by_value/zeros_likehuZU�B
�
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bBgradient_tape/binary_focal_crossentropy/clip_by_value/zeros_like_1huZU�B