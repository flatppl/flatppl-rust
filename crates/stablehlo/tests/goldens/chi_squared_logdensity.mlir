module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.multiply %0, %arg0 : tensor<f32>
    %6 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %7 = stablehlo.multiply %5, %6 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = chlo.lgamma %5 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.subtract %5, %3 : tensor<f32>
    %12 = stablehlo.log %4 : tensor<f32>
    %13 = stablehlo.multiply %11, %12 : tensor<f32>
    %14 = stablehlo.constant dense<2.0> : tensor<f32>
    %15 = stablehlo.divide %4, %14 : tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.add %8, %10 : tensor<f32>
    %18 = stablehlo.add %17, %13 : tensor<f32>
    %19 = stablehlo.add %18, %16 : tensor<f32>
    %20 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.select %2, %19, %21 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %22 : tensor<f32>
  }
}
