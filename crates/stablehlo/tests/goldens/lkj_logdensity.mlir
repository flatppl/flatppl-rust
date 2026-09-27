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
    %9 = stablehlo.reduce(%8 init: %6) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<2.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.subtract %arg0, %12 : tensor<f32>
    %14 = stablehlo.multiply %13, %11 : tensor<f32>
    %15 = stablehlo.multiply %10, %arg0 : tensor<f32>
    %16 = stablehlo.constant dense<0.0> : tensor<f32>
    %17 = stablehlo.add %15, %16 : tensor<f32>
    %18 = stablehlo.multiply %17, %10 : tensor<f32>
    %19 = stablehlo.constant dense<0.5> : tensor<f32>
    %20 = stablehlo.add %arg0, %19 : tensor<f32>
    %21 = stablehlo.multiply %10, %20 : tensor<f32>
    %22 = chlo.lgamma %20 : tensor<f32> -> tensor<f32>
    %23 = stablehlo.multiply %10, %22 : tensor<f32>
    %24 = chlo.lgamma %21 : tensor<f32> -> tensor<f32>
    %25 = stablehlo.subtract %23, %24 : tensor<f32>
    %26 = stablehlo.multiply %10, %25 : tensor<f32>
    %27 = stablehlo.constant dense<-1.0> : tensor<f32>
    %28 = stablehlo.add %15, %27 : tensor<f32>
    %29 = stablehlo.multiply %28, %12 : tensor<f32>
    %30 = stablehlo.add %18, %29 : tensor<f32>
    %31 = stablehlo.add %arg0, %16 : tensor<f32>
    %32 = stablehlo.multiply %10, %31 : tensor<f32>
    %33 = chlo.lgamma %31 : tensor<f32> -> tensor<f32>
    %34 = stablehlo.multiply %10, %33 : tensor<f32>
    %35 = chlo.lgamma %32 : tensor<f32> -> tensor<f32>
    %36 = stablehlo.subtract %34, %35 : tensor<f32>
    %37 = stablehlo.multiply %12, %36 : tensor<f32>
    %38 = stablehlo.add %26, %37 : tensor<f32>
    %39 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %40 = stablehlo.multiply %30, %39 : tensor<f32>
    %41 = stablehlo.add %40, %38 : tensor<f32>
    %42 = stablehlo.negate %41 : tensor<f32>
    %43 = stablehlo.add %14, %42 : tensor<f32>
    return %43 : tensor<f32>
  }
}
