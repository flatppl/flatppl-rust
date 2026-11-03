module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.3, 1.2]> : tensor<2xf32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %20 = stablehlo.constant dense<"0xC8DD4FC07849EEBF"> : tensor<2xf32>
    %21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.floor %22 : tensor<f32>
    %24 = stablehlo.abs %22 : tensor<f32>
    %25 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %26 = stablehlo.compare EQ, %24, %25 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %27 = stablehlo.constant dense<0.0> : tensor<f32>
    %28 = stablehlo.select %26, %27, %22 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.floor %28 : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %31 = stablehlo.subtract %20, %30 : tensor<2xf32>
    %32 = stablehlo.exponential %31 : tensor<2xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.reduce(%32 init: %33) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.select %26, %2, %27 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %36 = stablehlo.add %34, %35 : tensor<f32>
    %37 = stablehlo.log %36 : tensor<f32>
    %38 = stablehlo.add %37, %23 : tensor<f32>
    %39 = stablehlo.reduce(%0 init: %33) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %40 = stablehlo.log %39 : tensor<f32>
    %41 = stablehlo.subtract %38, %40 : tensor<f32>
    return %41 : tensor<f32>
  }
}
