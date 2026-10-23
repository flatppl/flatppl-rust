module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> tensor<f32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2xf32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.subtract %1, %0 : tensor<2xf32>
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %8 = stablehlo.divide %6, %7 : tensor<2xf32>
    %9 = stablehlo.constant dense<-0.5> : tensor<f32>
    %10 = stablehlo.reshape %8 : (tensor<2xf32>) -> tensor<2x1xf32>
    %11 = stablehlo.dot_general %10, %10, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2xf32>
    %12 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.multiply %12, %11 : tensor<2xf32>
    %14 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %16 = stablehlo.add %15, %13 : tensor<2xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%16 init: %17) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    return %18 : tensor<f32>
  }
}
