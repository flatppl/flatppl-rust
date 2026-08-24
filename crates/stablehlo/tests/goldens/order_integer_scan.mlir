module {
  func.func @logdensity(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %4 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%1: tensor<i32>, %2: tensor<i32>):
      %3 = stablehlo.maximum %1, %2 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) {
      window_dimensions = array<i64: 4>,
      window_strides = array<i64: 1>,
      padding = dense<[[3, 0]]> : tensor<1x2xi64>
    } : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
    %5 = stablehlo.constant dense<2147483647> : tensor<i32>
    %9 = "stablehlo.reduce_window"(%arg0, %5) ({
    ^bb0(%6: tensor<i32>, %7: tensor<i32>):
      %8 = stablehlo.minimum %6, %7 : tensor<i32>
      stablehlo.return %8 : tensor<i32>
    }) {
      window_dimensions = array<i64: 4>,
      window_strides = array<i64: 1>,
      padding = dense<[[3, 0]]> : tensor<1x2xi64>
    } : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
    return %4, %9 : tensor<4xi32>, tensor<4xi32>
  }
}
