module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %4 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %5 = stablehlo.constant dense<"0x1872313F"> : tensor<f32>
    %6 = stablehlo.log %arg1 : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %9 = stablehlo.subtract %4, %arg0 : tensor<f32>
    %10 = stablehlo.divide %9, %arg1 : tensor<f32>
    %11 = stablehlo.constant dense<-0.5> : tensor<f32>
    %12 = stablehlo.multiply %10, %10 : tensor<f32>
    %13 = stablehlo.multiply %11, %12 : tensor<f32>
    %14 = stablehlo.add %5, %7 : tensor<f32>
    %15 = stablehlo.add %14, %8 : tensor<f32>
    %16 = stablehlo.add %15, %13 : tensor<f32>
    return %16 : tensor<f32>
  }
}
