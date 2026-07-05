module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.log %arg1 : tensor<f32>
    %2 = stablehlo.negate %1 : tensor<f32>
    %3 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %4 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %5 = stablehlo.divide %4, %arg1 : tensor<f32>
    %6 = stablehlo.constant dense<-0.5> : tensor<f32>
    %7 = stablehlo.multiply %5, %5 : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.add %2, %3 : tensor<f32>
    %10 = stablehlo.add %9, %8 : tensor<f32>
    return %10 : tensor<f32>
  }
}
