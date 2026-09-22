module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.add %1, %arg0 : tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.constant dense<-0.6931471824645996> : tensor<f32>
    %10 = stablehlo.add %3, %5 : tensor<f32>
    %11 = stablehlo.add %10, %9 : tensor<f32>
    %12 = stablehlo.log %arg1 : tensor<f32>
    %13 = stablehlo.add %arg1, %6 : tensor<f32>
    %14 = stablehlo.log %13 : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.add %12, %15 : tensor<f32>
    %17 = stablehlo.multiply %arg0, %16 : tensor<f32>
    %18 = stablehlo.multiply %1, %14 : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.add %11, %17 : tensor<f32>
    %21 = stablehlo.add %20, %19 : tensor<f32>
    return %21 : tensor<f32>
  }
}
