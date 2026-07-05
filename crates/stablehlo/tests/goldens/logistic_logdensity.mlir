module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %2 = stablehlo.divide %1, %arg1 : tensor<f32>
    %3 = stablehlo.negate %2 : tensor<f32>
    %4 = stablehlo.log %arg1 : tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.exponential %3 : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.add %7, %6 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %10 = stablehlo.constant dense<2.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %3, %5 : tensor<f32>
    %14 = stablehlo.add %13, %12 : tensor<f32>
    return %14 : tensor<f32>
  }
}
