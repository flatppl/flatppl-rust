module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.2, 0.8]> : tensor<2xf32>
    %1 = stablehlo.constant dense<[-1.6094379425048828, -0.2231435328722]> : tensor<2xf32>
    %2 = stablehlo.constant dense<1.5> : tensor<f32>
    %3 = stablehlo.constant dense<[0.0, 1.5]> : tensor<2xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %5 = stablehlo.compare EQ, %4, %3 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %6 = stablehlo.constant dense<0.0> : tensor<f32>
    %7 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.select %5, %9, %10 : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %12 = stablehlo.add %1, %11 : tensor<2xf32>
    %13 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %16 = stablehlo.subtract %12, %15 : tensor<2xf32>
    %17 = stablehlo.exponential %16 : tensor<2xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.reduce(%17 init: %18) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.log %19 : tensor<f32>
    %21 = stablehlo.add %20, %14 : tensor<f32>
    %22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%0 init: %22) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.log %23 : tensor<f32>
    %25 = stablehlo.subtract %21, %24 : tensor<f32>
    return %25 : tensor<f32>
  }
}
