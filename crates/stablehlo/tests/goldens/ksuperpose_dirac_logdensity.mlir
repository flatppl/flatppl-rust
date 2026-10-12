module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.2, 0.8]> : tensor<2xf32>
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
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %18 = stablehlo.subtract %14, %17 : tensor<2xf32>
    %19 = stablehlo.exponential %18 : tensor<2xf32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%19 init: %20) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.log %21 : tensor<f32>
    %23 = stablehlo.add %22, %16 : tensor<f32>
    %24 = stablehlo.reduce(%0 init: %20) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.log %24 : tensor<f32>
    %26 = stablehlo.subtract %23, %25 : tensor<f32>
    return %26 : tensor<f32>
  }
}
