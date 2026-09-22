module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> tensor<f32> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %1 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %3 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %4 = stablehlo.select %2, %arg1, %3 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%4 init: %5) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<2.0> : tensor<f32>
    %8 = stablehlo.multiply %7, %arg0 : tensor<f32>
    %9 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.log %10 : tensor<f32>
    %12 = stablehlo.constant dense<-1.0> : tensor<f32>
    %13 = stablehlo.add %8, %12 : tensor<f32>
    %14 = stablehlo.multiply %13, %11 : tensor<f32>
    %15 = stablehlo.slice %6 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %16 = stablehlo.reshape %15 : (tensor<1xf32>) -> tensor<f32>
    %17 = stablehlo.log %16 : tensor<f32>
    %18 = stablehlo.constant dense<-2.0> : tensor<f32>
    %19 = stablehlo.add %8, %18 : tensor<f32>
    %20 = stablehlo.multiply %19, %17 : tensor<f32>
    %21 = stablehlo.add %14, %20 : tensor<f32>
    %22 = stablehlo.constant dense<0.0> : tensor<f32>
    %23 = stablehlo.add %8, %22 : tensor<f32>
    %24 = stablehlo.multiply %23, %7 : tensor<f32>
    %25 = stablehlo.constant dense<0.5> : tensor<f32>
    %26 = stablehlo.add %arg0, %25 : tensor<f32>
    %27 = stablehlo.multiply %7, %26 : tensor<f32>
    %28 = chlo.lgamma %26 : tensor<f32> -> tensor<f32>
    %29 = stablehlo.multiply %7, %28 : tensor<f32>
    %30 = chlo.lgamma %27 : tensor<f32> -> tensor<f32>
    %31 = stablehlo.subtract %29, %30 : tensor<f32>
    %32 = stablehlo.multiply %7, %31 : tensor<f32>
    %33 = stablehlo.constant dense<1.0> : tensor<f32>
    %34 = stablehlo.multiply %13, %33 : tensor<f32>
    %35 = stablehlo.add %24, %34 : tensor<f32>
    %36 = stablehlo.add %arg0, %22 : tensor<f32>
    %37 = stablehlo.multiply %7, %36 : tensor<f32>
    %38 = chlo.lgamma %36 : tensor<f32> -> tensor<f32>
    %39 = stablehlo.multiply %7, %38 : tensor<f32>
    %40 = chlo.lgamma %37 : tensor<f32> -> tensor<f32>
    %41 = stablehlo.subtract %39, %40 : tensor<f32>
    %42 = stablehlo.multiply %33, %41 : tensor<f32>
    %43 = stablehlo.add %32, %42 : tensor<f32>
    %44 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %45 = stablehlo.multiply %35, %44 : tensor<f32>
    %46 = stablehlo.add %45, %43 : tensor<f32>
    %47 = stablehlo.negate %46 : tensor<f32>
    %48 = stablehlo.add %21, %47 : tensor<f32>
    return %48 : tensor<f32>
  }
}
