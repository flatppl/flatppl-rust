module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %2 = stablehlo.log %arg1 : tensor<f32>
    %3 = stablehlo.negate %2 : tensor<f32>
    %4 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %5 = stablehlo.divide %4, %arg1 : tensor<f32>
    %6 = stablehlo.multiply %5, %5 : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.add %7, %6 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.add %1, %3 : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    return %12 : tensor<f32>
  }
}
