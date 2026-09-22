module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.add %1, %arg1 : tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %9 = stablehlo.constant dense<-0.6931471824645996> : tensor<f32>
    %10 = stablehlo.add %3, %5 : tensor<f32>
    %11 = stablehlo.add %10, %9 : tensor<f32>
    %12 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %13 = stablehlo.log %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.log %arg0 : tensor<f32>
    %16 = stablehlo.add %15, %14 : tensor<f32>
    %17 = stablehlo.multiply %1, %16 : tensor<f32>
    %18 = stablehlo.log %arg1 : tensor<f32>
    %19 = stablehlo.add %18, %14 : tensor<f32>
    %20 = stablehlo.multiply %arg1, %19 : tensor<f32>
    %21 = stablehlo.add %11, %17 : tensor<f32>
    %22 = stablehlo.add %21, %20 : tensor<f32>
    return %22 : tensor<f32>
  }
}
