module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<true> : tensor<i1>
    %4 = stablehlo.and %3, %3 : tensor<i1>
    %5 = stablehlo.select %4, %0, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = stablehlo.subtract %arg0, %2 : tensor<f32>
    %7 = stablehlo.log %5 : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.subtract %arg1, %2 : tensor<f32>
    %10 = stablehlo.subtract %2, %5 : tensor<f32>
    %11 = stablehlo.log %10 : tensor<f32>
    %12 = stablehlo.multiply %9, %11 : tensor<f32>
    %13 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %18 = chlo.lgamma %17 : tensor<f32> -> tensor<f32>
    %19 = stablehlo.add %8, %12 : tensor<f32>
    %20 = stablehlo.add %14, %16 : tensor<f32>
    %21 = stablehlo.add %20, %18 : tensor<f32>
    %22 = stablehlo.add %19, %21 : tensor<f32>
    %23 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %24 = stablehlo.negate %23 : tensor<f32>
    %25 = stablehlo.select %4, %22, %24 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %25 : tensor<f32>
  }
}
