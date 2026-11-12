module {
  func.func @logdensity() -> tensor<f32> {
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %21 = stablehlo.constant dense<"0xC8DD4FC01B6491C0"> : tensor<2xf32>
    %22 = stablehlo.optimization_barrier %21 : tensor<2xf32>
    %23 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %24 = stablehlo.reduce(%22 init: %23) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.floor %24 : tensor<f32>
    %26 = stablehlo.abs %24 : tensor<f32>
    %27 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %28 = stablehlo.compare EQ, %26, %27 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %29 = stablehlo.constant dense<0.0> : tensor<f32>
    %30 = stablehlo.select %28, %29, %24 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %31 = stablehlo.floor %30 : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %33 = stablehlo.subtract %22, %32 : tensor<2xf32>
    %34 = stablehlo.exponential %33 : tensor<2xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.reduce(%34 init: %35) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %37 = stablehlo.select %28, %4, %29 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %38 = stablehlo.add %36, %37 : tensor<f32>
    %39 = stablehlo.log %38 : tensor<f32>
    %40 = stablehlo.add %39, %25 : tensor<f32>
    %42 = stablehlo.constant dense<"0x1F99CF3E"> : tensor<f32>
    %43 = stablehlo.subtract %40, %42 : tensor<f32>
    return %43 : tensor<f32>
  }
}
