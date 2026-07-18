module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log %arg1 : tensor<f32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<f32>
    %7 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.constant dense<1.0> : tensor<f32>
    %10 = stablehlo.add %arg0, %9 : tensor<f32>
    %11 = stablehlo.log %4 : tensor<f32>
    %12 = stablehlo.multiply %10, %11 : tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.divide %arg1, %4 : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.add %6, %8 : tensor<f32>
    %17 = stablehlo.add %16, %13 : tensor<f32>
    %18 = stablehlo.add %17, %15 : tensor<f32>
    %19 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %20 = stablehlo.negate %19 : tensor<f32>
    %21 = stablehlo.select %2, %18, %20 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %21 : tensor<f32>
  }
}
