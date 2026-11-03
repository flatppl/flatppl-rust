module {
  func.func @logdensity() -> tensor<f32> {
    %1 = stablehlo.constant dense<"0x1002CEBFBD7F64BE"> : tensor<2xf32>
    %6 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %12 = stablehlo.select %6, %10, %11 : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %13 = stablehlo.reshape %12 : (tensor<2xf32>) -> tensor<2xf32>
    %14 = stablehlo.add %1, %13 : tensor<2xf32>
    %15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %16 = stablehlo.reduce(%14 init: %15) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.floor %16 : tensor<f32>
    %18 = stablehlo.abs %16 : tensor<f32>
    %19 = stablehlo.compare EQ, %18, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.constant dense<1.0> : tensor<f32>
    %21 = stablehlo.select %19, %7, %16 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.floor %21 : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %24 = stablehlo.subtract %14, %23 : tensor<2xf32>
    %25 = stablehlo.exponential %24 : tensor<2xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%25 init: %26) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %28 = stablehlo.select %19, %20, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.add %27, %28 : tensor<f32>
    %30 = stablehlo.log %29 : tensor<f32>
    %31 = stablehlo.add %30, %17 : tensor<f32>
    return %31 : tensor<f32>
  }
}
