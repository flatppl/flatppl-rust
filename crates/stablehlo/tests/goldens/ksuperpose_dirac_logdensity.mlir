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
    %15 = stablehlo.optimization_barrier %14 : tensor<2xf32>
    %16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.reduce(%15 init: %16) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.floor %17 : tensor<f32>
    %19 = stablehlo.abs %17 : tensor<f32>
    %20 = stablehlo.compare EQ, %19, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %21 = stablehlo.constant dense<1.0> : tensor<f32>
    %22 = stablehlo.select %20, %7, %17 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.floor %22 : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %25 = stablehlo.subtract %15, %24 : tensor<2xf32>
    %26 = stablehlo.exponential %25 : tensor<2xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.reduce(%26 init: %27) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.select %20, %21, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.add %28, %29 : tensor<f32>
    %31 = stablehlo.log %30 : tensor<f32>
    %32 = stablehlo.add %31, %18 : tensor<f32>
    return %32 : tensor<f32>
  }
}
