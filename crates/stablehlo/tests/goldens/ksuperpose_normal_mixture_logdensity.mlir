module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.3, 1.2]> : tensor<2xf32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %21 = stablehlo.constant dense<"0xC8DD4FC01B6491C0"> : tensor<2xf32>
    %22 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.floor %23 : tensor<f32>
    %25 = stablehlo.abs %23 : tensor<f32>
    %26 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %27 = stablehlo.compare EQ, %25, %26 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %28 = stablehlo.constant dense<0.0> : tensor<f32>
    %29 = stablehlo.select %27, %28, %23 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.floor %29 : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %32 = stablehlo.subtract %21, %31 : tensor<2xf32>
    %33 = stablehlo.exponential %32 : tensor<2xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.reduce(%33 init: %34) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %36 = stablehlo.select %27, %4, %28 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %37 = stablehlo.add %35, %36 : tensor<f32>
    %38 = stablehlo.log %37 : tensor<f32>
    %39 = stablehlo.add %38, %24 : tensor<f32>
    %40 = stablehlo.reduce(%0 init: %34) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %41 = stablehlo.log %40 : tensor<f32>
    %42 = stablehlo.subtract %39, %41 : tensor<f32>
    return %42 : tensor<f32>
  }
}
