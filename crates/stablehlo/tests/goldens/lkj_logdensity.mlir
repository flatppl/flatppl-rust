module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> tensor<f32> {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<3x3xf32>
    %1 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %2 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %3 = stablehlo.compare EQ, %1, %2 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %4 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %5 = stablehlo.select %3, %0, %4 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%5 init: %6) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %8 = stablehlo.log %7 : tensor<3xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<2.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.subtract %arg0, %13 : tensor<f32>
    %15 = stablehlo.multiply %14, %12 : tensor<f32>
    %16 = stablehlo.multiply %11, %arg0 : tensor<f32>
    %17 = stablehlo.constant dense<0.0> : tensor<f32>
    %18 = stablehlo.add %16, %17 : tensor<f32>
    %19 = stablehlo.multiply %18, %11 : tensor<f32>
    %20 = stablehlo.constant dense<0.5> : tensor<f32>
    %21 = stablehlo.add %arg0, %20 : tensor<f32>
    %22 = stablehlo.multiply %11, %21 : tensor<f32>
    %23 = chlo.lgamma %21 : tensor<f32> -> tensor<f32>
    %24 = stablehlo.multiply %11, %23 : tensor<f32>
    %25 = chlo.lgamma %22 : tensor<f32> -> tensor<f32>
    %26 = stablehlo.subtract %24, %25 : tensor<f32>
    %27 = stablehlo.multiply %11, %26 : tensor<f32>
    %28 = stablehlo.constant dense<-1.0> : tensor<f32>
    %29 = stablehlo.add %16, %28 : tensor<f32>
    %30 = stablehlo.multiply %29, %13 : tensor<f32>
    %31 = stablehlo.add %19, %30 : tensor<f32>
    %32 = stablehlo.add %arg0, %17 : tensor<f32>
    %33 = stablehlo.multiply %11, %32 : tensor<f32>
    %34 = chlo.lgamma %32 : tensor<f32> -> tensor<f32>
    %35 = stablehlo.multiply %11, %34 : tensor<f32>
    %36 = chlo.lgamma %33 : tensor<f32> -> tensor<f32>
    %37 = stablehlo.subtract %35, %36 : tensor<f32>
    %38 = stablehlo.multiply %13, %37 : tensor<f32>
    %39 = stablehlo.add %27, %38 : tensor<f32>
    %40 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %41 = stablehlo.multiply %31, %40 : tensor<f32>
    %42 = stablehlo.add %41, %39 : tensor<f32>
    %43 = stablehlo.negate %42 : tensor<f32>
    %44 = stablehlo.add %15, %43 : tensor<f32>
    return %44 : tensor<f32>
  }
}
