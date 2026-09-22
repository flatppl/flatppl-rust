module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.add %arg0, %1 : tensor<f32>
    %3 = stablehlo.multiply %0, %2 : tensor<f32>
    %4 = chlo.lgamma %3 : tensor<f32> -> tensor<f32>
    %5 = stablehlo.constant dense<3.141592653589793> : tensor<f32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.multiply %0, %7 : tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.multiply %0, %arg0 : tensor<f32>
    %11 = chlo.lgamma %10 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.constant dense<0.25> : tensor<f32>
    %14 = stablehlo.divide %13, %arg0 : tensor<f32>
    %15 = stablehlo.add %1, %14 : tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.multiply %3, %16 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %4, %9 : tensor<f32>
    %20 = stablehlo.add %19, %12 : tensor<f32>
    %21 = stablehlo.add %20, %18 : tensor<f32>
    return %21 : tensor<f32>
  }
}
