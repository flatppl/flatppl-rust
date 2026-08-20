module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%1: tensor<f32>, %2: tensor<f32>):
      %3 = stablehlo.add %1, %2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) {
      window_dimensions = array<i64: 4>,
      window_strides = array<i64: 1>,
      padding = dense<[[3, 0]]> : tensor<1x2xi64>
    } : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = "stablehlo.reduce_window"(%arg0, %5) ({
    ^bb0(%6: tensor<f32>, %7: tensor<f32>):
      %8 = stablehlo.multiply %6, %7 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }) {
      window_dimensions = array<i64: 4>,
      window_strides = array<i64: 1>,
      padding = dense<[[3, 0]]> : tensor<1x2xi64>
    } : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    return %4, %9 : tensor<4xf32>, tensor<4xf32>
  }
}
