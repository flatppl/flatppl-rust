module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.compare LT, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %3, %4 : tensor<i1>
    %6 = stablehlo.select %5, %0, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.subtract %arg0, %2 : tensor<f32>
    %8 = stablehlo.log %6 : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.subtract %arg1, %2 : tensor<f32>
    %11 = stablehlo.subtract %2, %6 : tensor<f32>
    %12 = stablehlo.log %11 : tensor<f32>
    %13 = stablehlo.multiply %10, %12 : tensor<f32>
    %14 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %19 = chlo.lgamma %18 : tensor<f32> -> tensor<f32>
    %20 = stablehlo.add %9, %13 : tensor<f32>
    %21 = stablehlo.add %15, %17 : tensor<f32>
    %22 = stablehlo.add %21, %19 : tensor<f32>
    %23 = stablehlo.add %20, %22 : tensor<f32>
    %24 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %25 = stablehlo.negate %24 : tensor<f32>
    %26 = stablehlo.select %5, %23, %25 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %26 : tensor<f32>
  }
}
