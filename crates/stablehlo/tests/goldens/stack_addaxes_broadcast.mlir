module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.constant dense<10.0> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.constant dense<2.0> : tensor<f32>
    %6 = stablehlo.constant dense<20.0> : tensor<f32>
    %7 = stablehlo.reshape %5 : (tensor<f32>) -> tensor<1xf32>
    %8 = stablehlo.reshape %6 : (tensor<f32>) -> tensor<1xf32>
    %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %10 = stablehlo.constant dense<3.0> : tensor<f32>
    %11 = stablehlo.constant dense<30.0> : tensor<f32>
    %12 = stablehlo.reshape %10 : (tensor<f32>) -> tensor<1xf32>
    %13 = stablehlo.reshape %11 : (tensor<f32>) -> tensor<1xf32>
    %14 = stablehlo.concatenate %12, %13, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %15 = stablehlo.reshape %4 : (tensor<2xf32>) -> tensor<1x2xf32>
    %16 = stablehlo.reshape %9 : (tensor<2xf32>) -> tensor<1x2xf32>
    %17 = stablehlo.reshape %14 : (tensor<2xf32>) -> tensor<1x2xf32>
    %18 = stablehlo.concatenate %15, %16, %17, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>
    %19 = stablehlo.constant dense<1.0> : tensor<f32>
    %20 = stablehlo.constant dense<2.0> : tensor<f32>
    %21 = stablehlo.constant dense<3.0> : tensor<f32>
    %22 = stablehlo.reshape %19 : (tensor<f32>) -> tensor<1xf32>
    %23 = stablehlo.reshape %20 : (tensor<f32>) -> tensor<1xf32>
    %24 = stablehlo.reshape %21 : (tensor<f32>) -> tensor<1xf32>
    %25 = stablehlo.concatenate %22, %23, %24, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %26 = stablehlo.reshape %25 : (tensor<3xf32>) -> tensor<3x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %28 = stablehlo.multiply %18, %27 : tensor<3x2xf32>
    return %28 : tensor<3x2xf32>
  }
}
