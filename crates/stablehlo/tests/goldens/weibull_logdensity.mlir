module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.log %arg0 : tensor<f32>
    %5 = stablehlo.log %arg1 : tensor<f32>
    %6 = stablehlo.negate %5 : tensor<f32>
    %7 = stablehlo.divide %0, %arg1 : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.subtract %arg0, %3 : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.power %7, %arg0 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %4, %6 : tensor<f32>
    %14 = stablehlo.add %13, %10 : tensor<f32>
    %15 = stablehlo.add %14, %12 : tensor<f32>
    return %15 : tensor<f32>
  }
}
