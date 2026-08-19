module {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    %1 = stablehlo.floor %arg0 : tensor<f32>
    %2 = stablehlo.ceil %1 : tensor<f32>
    %3 = stablehlo.multiply %arg0, %arg0 : tensor<f32>
    %4 = stablehlo.log %3 : tensor<f32>
    %5 = stablehlo.constant dense<2.302585092994046> : tensor<f32>
    %6 = stablehlo.divide %4, %5 : tensor<f32>
    %7 = stablehlo.multiply %arg0, %arg0 : tensor<f32>
    %8 = chlo.lgamma %7 : tensor<f32> -> tensor<f32>
    %9 = stablehlo.exponential %8 : tensor<f32>
    %10 = stablehlo.constant dense<2.0> : tensor<f32>
    %11 = stablehlo.minimum %arg0, %10 : tensor<f32>
    %12 = stablehlo.constant dense<-2.0> : tensor<f32>
    %13 = stablehlo.maximum %11, %12 : tensor<f32>
    return %0, %2, %6, %9, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
