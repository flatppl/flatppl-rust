module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.3, 1.2]> : tensor<2xf32>
    %21 = stablehlo.constant dense<"0xC8DD4FC01B6491C0"> : tensor<2xf32>
    %22 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.floor %23 : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %26 = stablehlo.subtract %21, %25 : tensor<2xf32>
    %27 = stablehlo.exponential %26 : tensor<2xf32>
    %28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.reduce(%27 init: %28) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.log %29 : tensor<f32>
    %31 = stablehlo.add %30, %24 : tensor<f32>
    %32 = stablehlo.reduce(%0 init: %28) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %33 = stablehlo.log %32 : tensor<f32>
    %34 = stablehlo.subtract %31, %33 : tensor<f32>
    return %34 : tensor<f32>
  }
}
