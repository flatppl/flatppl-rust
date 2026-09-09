module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<4.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.compare EQ, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %3, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.multiply %4, %7 : tensor<f32>
    %9 = stablehlo.negate %0 : tensor<f32>
    %10 = stablehlo.constant dense<1.0> : tensor<f32>
    %11 = stablehlo.add %4, %10 : tensor<f32>
    %12 = chlo.lgamma %11 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.add %8, %9 : tensor<f32>
    %15 = stablehlo.add %14, %13 : tensor<f32>
    %16 = stablehlo.compare EQ, %4, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.select %16, %9, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.select %5, %19, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.negate %21 : tensor<f32>
    %23 = stablehlo.select %2, %20, %22 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %23 : tensor<f32>
  }
}
