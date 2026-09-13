module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.3, 1.2]> : tensor<2xf32>
    %21 = stablehlo.constant dense<[-3.2479114532470703, -4.5434699058532715]> : tensor<2xf32>
    %22 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %25 = stablehlo.subtract %21, %24 : tensor<2xf32>
    %26 = stablehlo.exponential %25 : tensor<2xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.reduce(%26 init: %27) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.log %28 : tensor<f32>
    %30 = stablehlo.add %29, %23 : tensor<f32>
    %31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%0 init: %31) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %33 = stablehlo.log %32 : tensor<f32>
    %34 = stablehlo.subtract %30, %33 : tensor<f32>
    return %34 : tensor<f32>
  }
}
