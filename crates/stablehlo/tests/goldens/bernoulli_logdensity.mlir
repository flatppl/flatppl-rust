module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.multiply %2, %1 : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %6 = stablehlo.subtract %4, %5 : tensor<f32>
    %7 = stablehlo.subtract %4, %arg0 : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.multiply %6, %8 : tensor<f32>
    %10 = stablehlo.add %3, %9 : tensor<f32>
    return %10 : tensor<f32>
  }
}
