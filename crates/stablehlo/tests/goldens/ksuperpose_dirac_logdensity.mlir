module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.8> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.log %4 : tensor<2xf32>
    %6 = stablehlo.constant dense<1.5> : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.5> : tensor<f32>
    %9 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %12 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.compare EQ, %12, %11 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %14 = stablehlo.constant dense<0.0> : tensor<f32>
    %15 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %18 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %19 = stablehlo.select %13, %17, %18 : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %20 = stablehlo.add %5, %19 : tensor<2xf32>
    %21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<2xf32>
    %25 = stablehlo.exponential %24 : tensor<2xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%25 init: %26) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %28 = stablehlo.log %27 : tensor<f32>
    %29 = stablehlo.add %28, %22 : tensor<f32>
    %30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%4 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.log %31 : tensor<f32>
    %33 = stablehlo.subtract %29, %32 : tensor<f32>
    return %33 : tensor<f32>
  }
}
