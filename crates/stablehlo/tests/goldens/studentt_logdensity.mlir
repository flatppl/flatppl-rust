module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.add %arg0, %1 : tensor<f32>
    %4 = stablehlo.multiply %2, %3 : tensor<f32>
    %5 = chlo.lgamma %4 : tensor<f32> -> tensor<f32>
    %6 = stablehlo.constant dense<3.141592653589793> : tensor<f32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.multiply %2, %8 : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.multiply %2, %arg0 : tensor<f32>
    %12 = chlo.lgamma %11 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.multiply %0, %0 : tensor<f32>
    %15 = stablehlo.divide %14, %arg0 : tensor<f32>
    %16 = stablehlo.add %1, %15 : tensor<f32>
    %17 = stablehlo.log %16 : tensor<f32>
    %18 = stablehlo.multiply %4, %17 : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.add %5, %10 : tensor<f32>
    %21 = stablehlo.add %20, %13 : tensor<f32>
    %22 = stablehlo.add %21, %19 : tensor<f32>
    return %22 : tensor<f32>
  }
}
