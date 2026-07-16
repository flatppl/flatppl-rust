module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<4> : tensor<i32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %4 = stablehlo.log %3 : tensor<f32>
    %5 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %6 = stablehlo.multiply %5, %4 : tensor<f32>
    %7 = stablehlo.add %1, %6 : tensor<f32>
    return %7 : tensor<f32>
  }
}
