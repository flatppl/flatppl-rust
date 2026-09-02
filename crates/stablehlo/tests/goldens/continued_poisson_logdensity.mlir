module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<4.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log %0 : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.negate %0 : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.add %4, %8 : tensor<f32>
    %10 = chlo.lgamma %9 : tensor<f32> -> tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.add %6, %7 : tensor<f32>
    %13 = stablehlo.add %12, %11 : tensor<f32>
    %14 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.select %2, %13, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %16 : tensor<f32>
  }
}
