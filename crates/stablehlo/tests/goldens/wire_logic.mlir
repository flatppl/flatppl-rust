module {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = stablehlo.compare NE, %arg2, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.not %1 : tensor<i1>
    %3 = stablehlo.or %0, %2 : tensor<i1>
    %4 = stablehlo.abs %arg2 : tensor<f32>
    %5 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.abs %arg2 : tensor<f32>
    %8 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %9 = stablehlo.compare EQ, %7, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = stablehlo.xor %6, %9 : tensor<i1>
    %11 = stablehlo.and %3, %10 : tensor<i1>
    %12 = stablehlo.constant dense<10.0> : tensor<f32>
    %13 = stablehlo.constant dense<20.0> : tensor<f32>
    %14 = stablehlo.select %11, %12, %13 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %14 : tensor<f32>
  }
}
