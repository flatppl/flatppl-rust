module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.multiply %2, %1 : tensor<f32>
    %4 = stablehlo.negate %arg0 : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %7 = stablehlo.add %6, %5 : tensor<f32>
    %8 = chlo.lgamma %7 : tensor<f32> -> tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.add %3, %4 : tensor<f32>
    %11 = stablehlo.add %10, %9 : tensor<f32>
    return %11 : tensor<f32>
  }
}
