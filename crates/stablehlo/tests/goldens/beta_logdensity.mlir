module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.compare LT, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %3, %4 : tensor<i1>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.select %5, %0, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<f32>
    %10 = stablehlo.log %7 : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.subtract %arg1, %8 : tensor<f32>
    %13 = stablehlo.subtract %8, %7 : tensor<f32>
    %14 = stablehlo.log %13 : tensor<f32>
    %15 = stablehlo.multiply %12, %14 : tensor<f32>
    %16 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %21 = chlo.lgamma %20 : tensor<f32> -> tensor<f32>
    %22 = stablehlo.add %11, %15 : tensor<f32>
    %23 = stablehlo.add %17, %19 : tensor<f32>
    %24 = stablehlo.add %23, %21 : tensor<f32>
    %25 = stablehlo.add %22, %24 : tensor<f32>
    %26 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %27 = stablehlo.negate %26 : tensor<f32>
    %28 = stablehlo.select %5, %25, %27 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %28 : tensor<f32>
  }
}
