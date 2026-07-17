module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %6 = stablehlo.multiply %5, %arg0 : tensor<f32>
    %7 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = chlo.lgamma %6 : tensor<f32> -> tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.subtract %6, %12 : tensor<f32>
    %14 = stablehlo.log %4 : tensor<f32>
    %15 = stablehlo.multiply %13, %14 : tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.divide %4, %16 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %9, %11 : tensor<f32>
    %20 = stablehlo.add %19, %15 : tensor<f32>
    %21 = stablehlo.add %20, %18 : tensor<f32>
    %22 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %23 = stablehlo.negate %22 : tensor<f32>
    %24 = stablehlo.select %2, %21, %23 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %24 : tensor<f32>
  }
}
