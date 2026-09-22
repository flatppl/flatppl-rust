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
    %9 = stablehlo.subtract %arg0, %3 : tensor<f32>
    %10 = stablehlo.log %4 : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.multiply %arg1, %4 : tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.add %6, %8 : tensor<f32>
    %15 = stablehlo.add %14, %11 : tensor<f32>
    %16 = stablehlo.add %15, %13 : tensor<f32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.select %2, %16, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %19 : tensor<f32>
  }
}
