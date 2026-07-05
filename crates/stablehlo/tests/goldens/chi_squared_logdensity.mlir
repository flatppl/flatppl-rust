module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f32>
    %3 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %4 = stablehlo.multiply %2, %3 : tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.subtract %2, %8 : tensor<f32>
    %10 = stablehlo.log %0 : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.constant dense<2.0> : tensor<f32>
    %13 = stablehlo.divide %0, %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.add %5, %7 : tensor<f32>
    %16 = stablehlo.add %15, %11 : tensor<f32>
    %17 = stablehlo.add %16, %14 : tensor<f32>
    return %17 : tensor<f32>
  }
}
